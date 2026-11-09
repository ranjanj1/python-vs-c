# MJ: llama2.c (reference/run.c) inference in Mojo 1.2.0, standard library only.
# Single-threaded, float32 weights and activations, batch size 1.
# Speaks the line protocol in INTERFACE.md on stdin/stdout.

from std.memory.alloc import alloc, Layout
from std.sys import argv
from std.io.file_descriptor import RawFileDescriptor
from std.math import exp, cos, sin, sqrt, log

comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BPtr = Pointer[UInt8, MutUntrackedOrigin]
comptime IPtr = Pointer[Int, MutUntrackedOrigin]

comptime VW = 16  # SIMD width (float32 lanes) for the hot loops


def falloc(n: Int) -> FPtr:
    var p = alloc(Layout[Float32, alignment = .of_bytes[64]()](count=max(n, 1))).unsafe_leak()
    for i in range(n):
        p[unsafe_offset=i] = 0.0
    return p


def balloc(n: Int) -> BPtr:
    return alloc[UInt8]({count = max(n, 1)}).unsafe_leak()


def ialloc(n: Int) -> IPtr:
    var p = alloc[Int]({count = max(n, 1)}).unsafe_leak()
    for i in range(n):
        p[unsafe_offset=i] = 0
    return p


def read_fully(f: FileHandle, dst: BPtr, nbytes: Int) raises:
    var done = 0
    while done < nbytes:
        var chunk = min(nbytes - done, 1 << 28)
        var got = f.read(Span(unsafe_ptr=dst.unsafe_offset(done), length=chunk))
        if got <= 0:
            raise Error("short read")
        done += got


# ----------------------------------------------------------------------------
# neural net blocks


@always_inline
def dot(a: FPtr, b: FPtr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, VW](0)
    var j = 0
    while j + VW <= n:
        acc = a.unsafe_offset(j).unsafe_load[width=VW]().fma(
            b.unsafe_offset(j).unsafe_load[width=VW](), acc
        )
        j += VW
    var s = acc.reduce_add()
    while j < n:
        s += a[unsafe_offset=j] * b[unsafe_offset=j]
        j += 1
    return s


comptime MM_ROWS = 2  # rows of W per block (share each x load)
comptime MM_W = 32  # SIMD width of the matmul inner loop


def matmul(xout: FPtr, x: FPtr, w: FPtr, n: Int, d: Int):
    # W (d,n) @ x (n,) -> xout (d,); MM_ROWS rows at a time share each x load.
    comptime R = MM_ROWS
    comptime W = MM_W
    var i = 0
    while i + R <= d:
        var rb = w.unsafe_offset(i * n)
        var acc = Array[SIMD[DType.float32, W], R](fill=0)
        var j = 0
        while j + W <= n:
            var xv = x.unsafe_offset(j).unsafe_load[width=W]()
            comptime for r in range(R):
                acc[r] = rb.unsafe_offset(r * n + j).unsafe_load[width=W]().fma(xv, acc[r])
            j += W
        comptime for r in range(R):
            var s = acc[r].reduce_add()
            for jj in range(j, n):
                s += rb[unsafe_offset=r * n + jj] * x[unsafe_offset=jj]
            xout[unsafe_offset=i + r] = s
        i += R
    while i < d:
        xout[unsafe_offset=i] = dot(w.unsafe_offset(i * n), x, n)
        i += 1


def rmsnorm(o: FPtr, x: FPtr, weight: FPtr, size: Int):
    var ss = dot(x, x, size)
    ss /= Float32(size)
    ss += 1e-5
    ss = 1.0 / sqrt(ss)
    var j = 0
    while j + VW <= size:
        o.unsafe_offset(j).unsafe_store(
            weight.unsafe_offset(j).unsafe_load[width=VW]()
            * (ss * x.unsafe_offset(j).unsafe_load[width=VW]())
        )
        j += VW
    while j < size:
        o[unsafe_offset=j] = weight[unsafe_offset=j] * (ss * x[unsafe_offset=j])
        j += 1


def softmax(x: FPtr, size: Int):
    # max (order-independent), then exp and sum, then normalise (divide, as in C)
    var max_val = x[unsafe_offset=0]
    var i = 0
    if size >= VW:
        var mv = x.unsafe_load[width=VW]()
        i = VW
        while i + VW <= size:
            mv = max(mv, x.unsafe_offset(i).unsafe_load[width=VW]())
            i += VW
        max_val = mv.reduce_max()
    while i < size:
        if x[unsafe_offset=i] > max_val:
            max_val = x[unsafe_offset=i]
        i += 1
    var sv = SIMD[DType.float32, VW](0)
    i = 0
    while i + VW <= size:
        var e = exp(x.unsafe_offset(i).unsafe_load[width=VW]() - max_val)
        x.unsafe_offset(i).unsafe_store(e)
        sv += e
        i += VW
    var sum = sv.reduce_add()
    while i < size:
        var e = exp(x[unsafe_offset=i] - max_val)
        x[unsafe_offset=i] = e
        sum += e
        i += 1
    i = 0
    while i + VW <= size:
        x.unsafe_offset(i).unsafe_store(x.unsafe_offset(i).unsafe_load[width=VW]() / sum)
        i += VW
    while i < size:
        x[unsafe_offset=i] = x[unsafe_offset=i] / sum
        i += 1


def attention_all[HS: Int](
    q: FPtr, att: FPtr, kb: FPtr, vb: FPtr, kv_dim: Int, seq_len: Int, npos: Int,
    xb: FPtr, sqrt_hs: Float32, n_heads: Int, kv_mul: Int,
):
    # All heads, walking each cached K/V row contiguously (position-major).
    for t in range(npos):
        var krow = kb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            var qv = q.unsafe_offset(h * HS).unsafe_load[width=HS]()
            var kv = krow.unsafe_offset((h // kv_mul) * HS).unsafe_load[width=HS]()
            att[unsafe_offset=h * seq_len + t] = (qv * kv).reduce_add() / sqrt_hs
    for h in range(n_heads):
        softmax(att.unsafe_offset(h * seq_len), npos)
        xb.unsafe_offset(h * HS).unsafe_store(SIMD[DType.float32, HS](0))
    for t in range(npos):
        var vrow = vb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            var a = att[unsafe_offset=h * seq_len + t]
            var o = xb.unsafe_offset(h * HS)
            o.unsafe_store(
                vrow.unsafe_offset((h // kv_mul) * HS)
                .unsafe_load[width=HS]()
                .fma(SIMD[DType.float32, HS](a), o.unsafe_load[width=HS]())
            )


def attention_generic(
    q: FPtr, att: FPtr, kb: FPtr, vb: FPtr, kv_dim: Int, seq_len: Int, npos: Int,
    xb: FPtr, sqrt_hs: Float32, n_heads: Int, kv_mul: Int, hs: Int,
):
    # Any head size: same structure as attention_all, runtime-sized vectors.
    for t in range(npos):
        var krow = kb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            att[unsafe_offset=h * seq_len + t] = dot(q.unsafe_offset(h * hs), krow.unsafe_offset((h // kv_mul) * hs), hs) / sqrt_hs
    for h in range(n_heads):
        softmax(att.unsafe_offset(h * seq_len), npos)
    for i in range(n_heads * hs):
        xb[unsafe_offset=i] = 0.0
    for t in range(npos):
        var vrow = vb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            var a = att[unsafe_offset=h * seq_len + t]
            var o = xb.unsafe_offset(h * hs)
            var v = vrow.unsafe_offset((h // kv_mul) * hs)
            var j = 0
            while j + VW <= hs:
                o.unsafe_offset(j).unsafe_store(
                    v.unsafe_offset(j).unsafe_load[width=VW]().fma(
                        SIMD[DType.float32, VW](a), o.unsafe_offset(j).unsafe_load[width=VW]()
                    )
                )
                j += VW
            while j < hs:
                o[unsafe_offset=j] = o[unsafe_offset=j] + a * v[unsafe_offset=j]
                j += 1


def add_into(x: FPtr, y: FPtr, n: Int):
    var j = 0
    while j + VW <= n:
        x.unsafe_offset(j).unsafe_store(
            x.unsafe_offset(j).unsafe_load[width=VW]()
            + y.unsafe_offset(j).unsafe_load[width=VW]()
        )
        j += VW
    while j < n:
        x[unsafe_offset=j] = x[unsafe_offset=j] + y[unsafe_offset=j]
        j += 1


# ----------------------------------------------------------------------------
# Transformer


struct Model(Movable):
    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var kv_dim: Int
    var kv_mul: Int
    # weights
    var token_embedding_table: FPtr
    var rms_att_weight: FPtr
    var wq: FPtr
    var wk: FPtr
    var wv: FPtr
    var wo: FPtr
    var rms_ffn_weight: FPtr
    var w1: FPtr
    var w2: FPtr
    var w3: FPtr
    var rms_final_weight: FPtr
    var wcls: FPtr
    # run state
    var x: FPtr
    var xb: FPtr
    var xb2: FPtr
    var hb: FPtr
    var hb2: FPtr
    var q: FPtr
    var att: FPtr
    var logits: FPtr
    var key_cache: FPtr
    var value_cache: FPtr
    var inv_freq: FPtr  # (head_size/2,) RoPE frequencies, as computed by run.c
    var rope_cos: FPtr  # (head_size/2,) for the current position
    var rope_sin: FPtr

    def __init__(out self, path: String) raises:
        var f = open(path, "r")
        var hdr = balloc(28)
        read_fully(f, hdr, 28)
        var hp = hdr.unsafe_bitcast[Int32]()
        self.dim = Int(hp[unsafe_offset=0])
        self.hidden_dim = Int(hp[unsafe_offset=1])
        self.n_layers = Int(hp[unsafe_offset=2])
        self.n_heads = Int(hp[unsafe_offset=3])
        self.n_kv_heads = Int(hp[unsafe_offset=4])
        var vs = Int(hp[unsafe_offset=5])
        self.seq_len = Int(hp[unsafe_offset=6])
        var shared = vs > 0
        self.vocab_size = vs if vs > 0 else -vs
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.dim * self.n_kv_heads) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads

        var file_size = Int(f.seek(0, 2))
        _ = f.seek(28, 0)
        var nbytes = file_size - 28
        var nfloats = nbytes // 4
        var data = alloc(
            Layout[Float32, alignment = .of_bytes[64]()](count=nfloats + 1)
        ).unsafe_leak()
        read_fully(f, data.unsafe_bitcast[UInt8](), nfloats * 4)
        f.close()

        var dim = self.dim
        var hd = self.hidden_dim
        var nl = self.n_layers
        var hs = self.head_size
        var ptr = data
        self.token_embedding_table = ptr
        ptr = ptr.unsafe_offset(self.vocab_size * dim)
        self.rms_att_weight = ptr
        ptr = ptr.unsafe_offset(nl * dim)
        self.wq = ptr
        ptr = ptr.unsafe_offset(nl * dim * (self.n_heads * hs))
        self.wk = ptr
        ptr = ptr.unsafe_offset(nl * dim * (self.n_kv_heads * hs))
        self.wv = ptr
        ptr = ptr.unsafe_offset(nl * dim * (self.n_kv_heads * hs))
        self.wo = ptr
        ptr = ptr.unsafe_offset(nl * (self.n_heads * hs) * dim)
        self.rms_ffn_weight = ptr
        ptr = ptr.unsafe_offset(nl * dim)
        self.w1 = ptr
        ptr = ptr.unsafe_offset(nl * dim * hd)
        self.w2 = ptr
        ptr = ptr.unsafe_offset(nl * hd * dim)
        self.w3 = ptr
        ptr = ptr.unsafe_offset(nl * dim * hd)
        self.rms_final_weight = ptr
        ptr = ptr.unsafe_offset(dim)
        ptr = ptr.unsafe_offset(self.seq_len * hs // 2)
        ptr = ptr.unsafe_offset(self.seq_len * hs // 2)
        self.wcls = self.token_embedding_table if shared else ptr

        self.x = falloc(dim)
        self.xb = falloc(dim)
        self.xb2 = falloc(dim)
        self.hb = falloc(hd)
        self.hb2 = falloc(hd)
        self.q = falloc(dim)
        self.key_cache = falloc(nl * self.seq_len * self.kv_dim)
        self.value_cache = falloc(nl * self.seq_len * self.kv_dim)
        self.att = falloc(self.n_heads * self.seq_len)
        self.logits = falloc(self.vocab_size)

        # freq = 1.0f / powf(10000.0f, head_dim / (float)head_size).
        # The exponent is formed in float32 exactly as in C; the power itself
        # is evaluated in float64 and rounded, which reproduces a correctly
        # rounded powf (Mojo's float32 pow is exp(y*log(x)) and less exact).
        var half = hs // 2
        self.inv_freq = falloc(half)
        self.rope_cos = falloc(half)
        self.rope_sin = falloc(half)
        for k in range(half):
            var e32 = Float32(2 * k) / Float32(hs)
            var p64 = exp(Float64(e32) * log(Float64(10000.0)))
            var p32 = Float32(p64)
            self.inv_freq[unsafe_offset=k] = Float32(1.0) / p32

    def forward(mut self, token: Int, pos: Int):
        var dim = self.dim
        var kv_dim = self.kv_dim
        var kv_mul = self.kv_mul
        var hidden_dim = self.hidden_dim
        var head_size = self.head_size
        var seq_len = self.seq_len
        var x = self.x

        # copy the token embedding into x
        var row = self.token_embedding_table.unsafe_offset(token * dim)
        for i in range(dim):
            x[unsafe_offset=i] = row[unsafe_offset=i]

        # RoPE rotation factors for this position (identical for every layer)
        var half = head_size // 2
        for k in range(half):
            var val = Float32(pos) * self.inv_freq[unsafe_offset=k]
            self.rope_cos[unsafe_offset=k] = cos(val)
            self.rope_sin[unsafe_offset=k] = sin(val)

        var sqrt_hs = sqrt(Float32(head_size))

        for l in range(self.n_layers):
            rmsnorm(self.xb, x, self.rms_att_weight.unsafe_offset(l * dim), dim)

            var loff = l * seq_len * kv_dim
            var k = self.key_cache.unsafe_offset(loff + pos * kv_dim)
            var v = self.value_cache.unsafe_offset(loff + pos * kv_dim)

            matmul(self.q, self.xb, self.wq.unsafe_offset(l * dim * dim), dim, dim)
            matmul(k, self.xb, self.wk.unsafe_offset(l * dim * kv_dim), dim, kv_dim)
            matmul(v, self.xb, self.wv.unsafe_offset(l * dim * kv_dim), dim, kv_dim)

            # RoPE: rotate q and k in each head
            var i = 0
            while i < dim:
                var hdi = (i % head_size) // 2
                var fcr = self.rope_cos[unsafe_offset=hdi]
                var fci = self.rope_sin[unsafe_offset=hdi]
                var q0 = self.q[unsafe_offset=i]
                var q1 = self.q[unsafe_offset=i + 1]
                self.q[unsafe_offset=i] = q0 * fcr - q1 * fci
                self.q[unsafe_offset=i + 1] = q0 * fci + q1 * fcr
                if i < kv_dim:
                    var k0 = k[unsafe_offset=i]
                    var k1 = k[unsafe_offset=i + 1]
                    k[unsafe_offset=i] = k0 * fcr - k1 * fci
                    k[unsafe_offset=i + 1] = k0 * fci + k1 * fcr
                i += 2

            # multihead attention (compile-time head size fast path, else generic)
            var kbase = self.key_cache.unsafe_offset(loff)
            var vbase = self.value_cache.unsafe_offset(loff)
            var npos = pos + 1
            if head_size == 64:
                attention_all[64](self.q, self.att, kbase, vbase, kv_dim, seq_len, npos, self.xb, sqrt_hs, self.n_heads, kv_mul)
            elif head_size == 128:
                attention_all[128](self.q, self.att, kbase, vbase, kv_dim, seq_len, npos, self.xb, sqrt_hs, self.n_heads, kv_mul)
            elif head_size == 32:
                attention_all[32](self.q, self.att, kbase, vbase, kv_dim, seq_len, npos, self.xb, sqrt_hs, self.n_heads, kv_mul)
            elif head_size == 16:
                attention_all[16](self.q, self.att, kbase, vbase, kv_dim, seq_len, npos, self.xb, sqrt_hs, self.n_heads, kv_mul)
            else:
                attention_generic(self.q, self.att, kbase, vbase, kv_dim, seq_len, npos, self.xb, sqrt_hs, self.n_heads, kv_mul, head_size)

            matmul(self.xb2, self.xb, self.wo.unsafe_offset(l * dim * dim), dim, dim)
            add_into(x, self.xb2, dim)

            rmsnorm(self.xb, x, self.rms_ffn_weight.unsafe_offset(l * dim), dim)
            matmul(self.hb, self.xb, self.w1.unsafe_offset(l * dim * hidden_dim), dim, hidden_dim)
            matmul(self.hb2, self.xb, self.w3.unsafe_offset(l * dim * hidden_dim), dim, hidden_dim)

            # SwiGLU
            var j = 0
            while j + VW <= hidden_dim:
                var hv = self.hb.unsafe_offset(j).unsafe_load[width=VW]()
                hv = hv * (1.0 / (1.0 + exp(-hv)))
                hv = hv * self.hb2.unsafe_offset(j).unsafe_load[width=VW]()
                self.hb.unsafe_offset(j).unsafe_store(hv)
                j += VW
            while j < hidden_dim:
                var hv = self.hb[unsafe_offset=j]
                hv = hv * (1.0 / (1.0 + exp(-hv)))
                hv = hv * self.hb2[unsafe_offset=j]
                self.hb[unsafe_offset=j] = hv
                j += 1

            matmul(self.xb, self.hb, self.w2.unsafe_offset(l * dim * hidden_dim), hidden_dim, dim)
            add_into(x, self.xb, dim)

        rmsnorm(x, x, self.rms_final_weight, dim)
        matmul(self.logits, x, self.wcls, dim, self.vocab_size)

    def argmax(self) -> Int:
        var max_i = 0
        var max_p = self.logits[unsafe_offset=0]
        for i in range(1, self.vocab_size):
            var p = self.logits[unsafe_offset=i]
            if p > max_p:
                max_i = i
                max_p = p
        return max_i


# ----------------------------------------------------------------------------
# BPE tokenizer (byte strings, exact-match hash lookup replaces bsearch)


def hash_bytes(p: BPtr, n: Int) -> UInt64:
    var h: UInt64 = 1469598103934665603
    for i in range(n):
        h = (h ^ UInt64(p[unsafe_offset=i])) * 1099511628211
    return h


def bytes_eq(a: BPtr, b: BPtr, n: Int) -> Bool:
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            return False
    return True


struct Tokenizer(Movable):
    var vocab_size: Int
    var max_token_length: Int
    var scores: FPtr
    var offs: IPtr  # byte offset of each piece in `buf`
    var lens: IPtr  # length of each piece (up to its first NUL, like a C string)
    var buf: BPtr
    var table: IPtr  # open-addressing table of id+1, 0 = empty
    var mask: Int
    var scratch: BPtr

    def __init__(out self, path: String, vocab_size: Int) raises:
        self.vocab_size = vocab_size
        var f = open(path, "r")
        var size = Int(f.seek(0, 2))
        _ = f.seek(0, 0)
        var raw = balloc(size)
        read_fully(f, raw, size)
        f.close()
        self.max_token_length = Int(raw.unsafe_bitcast[Int32]()[unsafe_offset=0])
        self.scores = falloc(vocab_size)
        self.offs = ialloc(vocab_size)
        self.lens = ialloc(vocab_size)
        self.buf = raw
        var o = 4
        for i in range(vocab_size):
            if o + 8 > size:
                raise Error("failed read")
            self.scores[unsafe_offset=i] = (raw.unsafe_offset(o)).unsafe_bitcast[Float32]()[unsafe_offset=0]
            var ln = Int(raw.unsafe_offset(o + 4).unsafe_bitcast[Int32]()[unsafe_offset=0])
            o += 8
            if o + ln > size:
                raise Error("failed read")
            self.offs[unsafe_offset=i] = o
            var cl = 0
            while cl < ln and raw[unsafe_offset=o + cl] != 0:
                cl += 1
            self.lens[unsafe_offset=i] = cl
            o += ln
        var cap = 1
        while cap < vocab_size * 2:
            cap *= 2
        self.mask = cap - 1
        self.table = ialloc(cap)
        var longest = self.max_token_length
        for i in range(vocab_size):
            longest = max(longest, self.lens[unsafe_offset=i])
        self.scratch = balloc(longest * 2 + 8)
        # insert in id order; on duplicate strings keep the first (any match is valid)
        for i in range(vocab_size):
            var p = self.buf.unsafe_offset(self.offs[unsafe_offset=i])
            var n = self.lens[unsafe_offset=i]
            var slot = Int(hash_bytes(p, n)) & self.mask
            var dup = False
            while self.table[unsafe_offset=slot] != 0:
                var j = self.table[unsafe_offset=slot] - 1
                if self.lens[unsafe_offset=j] == n and bytes_eq(self.buf.unsafe_offset(self.offs[unsafe_offset=j]), p, n):
                    dup = True
                    break
                slot = (slot + 1) & self.mask
            if not dup:
                self.table[unsafe_offset=slot] = i + 1

    def lookup(self, p: BPtr, n: Int) -> Int:
        var slot = Int(hash_bytes(p, n)) & self.mask
        while self.table[unsafe_offset=slot] != 0:
            var j = self.table[unsafe_offset=slot] - 1
            if self.lens[unsafe_offset=j] == n and bytes_eq(self.buf.unsafe_offset(self.offs[unsafe_offset=j]), p, n):
                return j
            slot = (slot + 1) & self.mask
        return -1

    def piece(self, id: Int) -> BPtr:
        return self.buf.unsafe_offset(self.offs[unsafe_offset=id])

    def encode(self, line: List[UInt8], start: Int) -> List[Int]:
        # C strings: the text ends at its first NUL byte
        var text_len = 0
        while start + text_len < len(line) and line[start + text_len] != 0:
            text_len += 1
        var text = balloc(text_len + 1)
        for i in range(text_len):
            text[unsafe_offset=i] = line[start + i]
        var tokens = List[Int](capacity=text_len + 3)
        tokens.append(1)  # BOS
        if text_len > 0:
            var sp = balloc(1)
            sp[unsafe_offset=0] = 32
            tokens.append(self.lookup(sp, 1))
            sp.unsafe_free()
        var sb = balloc(8)
        var str_len = 0
        for ci in range(text_len):
            var c = text[unsafe_offset=ci]
            if (c & 0xC0) != 0x80:
                str_len = 0
            sb[unsafe_offset=str_len] = c
            str_len += 1
            var nxt: UInt8 = text[unsafe_offset=ci + 1] if ci + 1 < text_len else 0
            if (nxt & 0xC0) == 0x80 and str_len < 4:
                continue
            var id = self.lookup(sb, str_len)
            if id != -1:
                tokens.append(id)
            else:
                for i in range(str_len):
                    tokens.append(Int(sb[unsafe_offset=i]) + 3)
            str_len = 0
        sb.unsafe_free()
        text.unsafe_free()

        # merge the best consecutive pair each iteration
        var sc = self.scratch
        while True:
            var best_score: Float32 = -1e10
            var best_id = -1
            var best_idx = -1
            for i in range(len(tokens) - 1):
                var a = tokens[i]
                var b = tokens[i + 1]
                var la = self.lens[unsafe_offset=a]
                var lb = self.lens[unsafe_offset=b]
                var pa = self.piece(a)
                var pb = self.piece(b)
                for k in range(la):
                    sc[unsafe_offset=k] = pa[unsafe_offset=k]
                for k in range(lb):
                    sc[unsafe_offset=la + k] = pb[unsafe_offset=k]
                var id = self.lookup(sc, la + lb)
                if id != -1 and self.scores[unsafe_offset=id] > best_score:
                    best_score = self.scores[unsafe_offset=id]
                    best_id = id
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            _ = tokens.pop(best_idx + 1)
        return tokens^

    def decode_into(self, prev_token: Int, token: Int, mut out_bytes: List[UInt8]):
        var p = self.piece(token)
        var n = self.lens[unsafe_offset=token]
        if prev_token == 1 and n > 0 and p[unsafe_offset=0] == 32:
            p = p.unsafe_offset(1)
            n -= 1
        # sscanf(piece, "<0x%02hhX>", &byte_val): "<0x" then 1-2 hex digits
        if n >= 4 and p[unsafe_offset=0] == 60 and p[unsafe_offset=1] == 48 and p[unsafe_offset=2] == 120:
            var h0 = hexval(p[unsafe_offset=3])
            if h0 >= 0:
                var bv = h0
                if n >= 5:
                    var h1 = hexval(p[unsafe_offset=4])
                    if h1 >= 0:
                        bv = h0 * 16 + h1
                # a single byte piece; byte 0 is an empty C string
                if bv != 0:
                    out_bytes.append(UInt8(bv))
                return
        for i in range(n):
            out_bytes.append(p[unsafe_offset=i])


def hexval(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 65 and c <= 70:
        return Int(c) - 55
    if c >= 97 and c <= 102:
        return Int(c) - 87
    return -1


# ----------------------------------------------------------------------------
# stdin line reader (raw bytes, unbuffered by libc so nothing is lost)


struct LineReader(Movable):
    var fd: RawFileDescriptor
    var buf: BPtr
    var cap: Int
    var start: Int
    var end: Int
    var eof: Bool

    def __init__(out self):
        self.fd = RawFileDescriptor(0)
        self.cap = 1 << 16
        self.buf = balloc(self.cap)
        self.start = 0
        self.end = 0
        self.eof = False

    def readline(mut self, mut line: List[UInt8]) raises -> Bool:
        line.clear()
        while True:
            while self.start < self.end:
                var c = self.buf[unsafe_offset=self.start]
                self.start += 1
                if c == 10:
                    return True
                line.append(c)
            if self.eof:
                return len(line) > 0
            var n = self.fd.read_bytes(Span(unsafe_ptr=self.buf, length=self.cap))
            self.start = 0
            self.end = n
            if n <= 0:
                self.end = 0
                self.eof = True


# ----------------------------------------------------------------------------
# protocol helpers


def emit(s: String):
    print(s, flush=True)


def starts_with(line: List[UInt8], word: StaticString) -> Bool:
    var wp = word.unsafe_ptr()
    var n = word.byte_length()
    if len(line) < n:
        return False
    for i in range(n):
        if line[i] != wp[unsafe_offset=i]:
            return False
    return True


def parse_ints(line: List[UInt8], start: Int) -> List[Int]:
    var out = List[Int]()
    var i = start
    var n = len(line)
    while i < n:
        while i < n and (line[i] == 32 or line[i] == 9 or line[i] == 13):
            i += 1
        if i >= n:
            break
        var neg = False
        if line[i] == 45:
            neg = True
            i += 1
        var v = 0
        while i < n and line[i] >= 48 and line[i] <= 57:
            v = v * 10 + Int(line[i]) - 48
            i += 1
        while i < n and line[i] != 32 and line[i] != 9:
            i += 1
        out.append(-v if neg else v)
    return out^


def bytes_to_string(line: List[UInt8], start: Int, end: Int) -> String:
    var s = String()
    var p = line.unsafe_ptr()
    s += StringSlice(unsafe_from_utf8=Span(unsafe_ptr=p.unsafe_offset(start), length=end - start))
    return s^


comptime HEXD = "0123456789abcdef"


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: mj <checkpoint.bin> <tokenizer.bin>")
        return
    var model = Model(String(args[1]))
    var tok = Tokenizer(String(args[2]), model.vocab_size)
    var reader = LineReader()
    var line = List[UInt8]()
    var hexp = HEXD.ptr()
    emit("READY")

    while reader.readline(line):
        # tolerate CRLF
        if len(line) > 0 and line[len(line) - 1] == 13:
            _ = line.pop()
        if starts_with(line, "QUIT"):
            return
        elif starts_with(line, "ENCODE"):
            var st = min(7, len(line))
            var ids = tok.encode(line, st)
            var s = String("IDS")
            for i in range(len(ids)):
                s += " "
                s += String(ids[i])
            emit(s)
        elif starts_with(line, "PIECE"):
            var v = parse_ints(line, 5)
            var bytes = List[UInt8]()
            if len(v) >= 2:
                tok.decode_into(v[0], v[1], bytes)
            var s = String("HEX ")
            for i in range(len(bytes)):
                var b = Int(bytes[i])
                s += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=hexp.unsafe_offset((b >> 4)), length=1)))
                s += String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=hexp.unsafe_offset((b & 15)), length=1)))
            emit(s)
        elif starts_with(line, "GENERATE"):
            # GENERATE <max_new> <text>
            var i = 9
            var n = len(line)
            var max_new = 0
            while i < n and line[i] >= 48 and line[i] <= 57:
                max_new = max_new * 10 + Int(line[i]) - 48
                i += 1
            if i < n and line[i] == 32:
                i += 1
            var st = min(i, n)
            var prompt = tok.encode(line, st)
            var np = len(prompt)
            var pos = 0
            var token = prompt[0]
            var count = 0
            var reason: String
            while True:
                if count >= max_new:
                    reason = "max"
                    break
                if pos >= model.seq_len:
                    reason = "context"
                    break
                model.forward(token, pos)
                var nxt: Int
                var forced = pos < np - 1
                if forced:
                    nxt = prompt[pos + 1]
                else:
                    nxt = model.argmax()
                pos += 1
                if nxt == 1:
                    reason = "stop"
                    break
                if not forced:
                    emit("T " + String(nxt))
                    count += 1
                token = nxt
            emit("DONE " + String(count) + " " + reason)
        elif starts_with(line, "FIXED"):
            var ids = parse_ints(line, 5)
            var n = min(len(ids), model.seq_len)
            var s = String("DONE ") + String(n)
            for p in range(n):
                model.forward(ids[p], p)
                s += " "
                s += String(model.argmax())
            emit(s)
        elif starts_with(line, "LOGITS"):
            # LOGITS <path> <id> ...
            var i = 7
            var nl = len(line)
            var ps = i
            while i < nl and line[i] != 32:
                i += 1
            var path = bytes_to_string(line, ps, i)
            var ids = parse_ints(line, i)
            var n = min(len(ids), model.seq_len)
            var f = open(path, "w")
            var lb = model.logits.unsafe_bitcast[UInt8]()
            for p in range(n):
                model.forward(ids[p], p)
                f.write_bytes(Span(unsafe_ptr=lb, length=model.vocab_size * 4))
            f.close()
            emit("DONE " + String(n))
        else:
            emit("ERR unknown command")
