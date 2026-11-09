#!/usr/bin/env python3
"""P1 port of reference/run.c (llama2.c) -- Python standard library only, scalar loops.

Usage: python3 run.py <checkpoint.bin> <tokenizer.bin>
Speaks the line protocol described in INTERFACE.md on stdin/stdout.

Precision: all weights, activations, KV cache and logits are stored as float32
(array('f')); every vector produced by an operation is rounded to float32 when
stored, as in the C reference.  Inner dot-product accumulators are Python
floats (the only scalar float type in the standard library), see NOTES.md.
"""
import sys
import math
import struct
from array import array

_F32 = struct.Struct('<f')


def f32(v):
    """Round a Python float to the nearest float32 value."""
    return _F32.unpack(_F32.pack(v))[0]


def rf32(values):
    """Round a sequence of Python floats to float32, returned as a list."""
    return array('f', values).tolist()


# ---------------------------------------------------------------------------
# Transformer

class Transformer:
    def __init__(self, path):
        with open(path, 'rb') as f:
            hdr = struct.unpack('<7i', f.read(28))
            (self.dim, self.hidden_dim, self.n_layers, self.n_heads,
             self.n_kv_heads, vocab_size, self.seq_len) = hdr
            shared = vocab_size > 0
            self.vocab_size = abs(vocab_size)
            # read the weights straight into a pre-sized float32 array (no temporary
            # bytes object: halves peak memory and load time)
            f.seek(0, 2)
            nfl = (f.tell() - 28) // 4
            f.seek(28)
            # (readinto on the array itself: PyPy rejects sliced/cast memoryviews here)
            data = array('f', [0.0]) * nfl
            got = f.readinto(data) or 0
            if got != 4 * nfl:          # short read: fall back to a plain fromfile
                f.seek(28)
                data = array('f')
                data.fromfile(f, nfl)
        if sys.byteorder != 'little':
            data.byteswap()
        p = self
        dim, hd, L, V = p.dim, p.hidden_dim, p.n_layers, p.vocab_size
        head_size = dim // p.n_heads
        kv_dim = dim * p.n_kv_heads // p.n_heads
        self.head_size = head_size
        self.kv_dim = kv_dim
        self.kv_mul = p.n_heads // p.n_kv_heads
        self.data = data
        # offsets into data (in floats), same layout as memory_map_weights()
        off = 0
        self.o_emb = off; off += V * dim
        self.o_rms_att = off; off += L * dim
        self.o_wq = off; off += L * dim * (p.n_heads * head_size)
        self.o_wk = off; off += L * dim * (p.n_kv_heads * head_size)
        self.o_wv = off; off += L * dim * (p.n_kv_heads * head_size)
        self.o_wo = off; off += L * (p.n_heads * head_size) * dim
        self.o_rms_ffn = off; off += L * dim
        self.o_w1 = off; off += L * dim * hd
        self.o_w2 = off; off += L * hd * dim
        self.o_w3 = off; off += L * dim * hd
        self.o_rms_final = off; off += dim
        off += p.seq_len * head_size // 2  # freq_cis_real (unused)
        off += p.seq_len * head_size // 2  # freq_cis_imag (unused)
        self.o_wcls = self.o_emb if shared else off
        if not shared and off + V * dim > len(data):
            raise ValueError('checkpoint too small for unshared classifier')

        # KV cache: (layer, seq_len, kv_dim) float32, allocated once
        self.kv_size = L * p.seq_len * kv_dim
        self.reset()

        # RoPE: freq per even index i, as 1.0f / powf(10000.0f, head_dim / (float)head_size)
        self.rope_freq = []
        for i in range(0, dim, 2):
            head_dim = i % head_size
            e = f32(head_dim / head_size)
            self.rope_freq.append(f32(1.0 / f32(10000.0 ** e)))
        self.inv_sqrt_hs = f32(math.sqrt(head_size))

    def reset(self):
        """Clean KV cache (each command starts from position 0)."""
        self.key_cache = array('f', [0.0]) * self.kv_size
        self.value_cache = array('f', [0.0]) * self.kv_size

    # -- blocks ---------------------------------------------------------------

    def matmul(self, x, off, n, d):
        """W (d,n) @ x (n,) -> (d,) float32, W stored row-major at data[off:]."""
        return rf32(dots(self.data, off, n, 1, n, d, x))

    def rmsnorm(self, x, off):
        size = len(x)
        ss = 0.0
        for v in x:
            ss += v * v
        ss = f32(ss)
        ss = f32(ss / size)
        ss = f32(ss + f32(1e-5))
        ss = f32(1.0 / f32(math.sqrt(ss)))
        t = rf32([ss * v for v in x])                       # (ss * x[j])
        return rf32([a * b for a, b in zip(self.data[off:off + size], t)])  # weight[j] * ...

    def forward(self, token, pos):
        dim, hd = self.dim, self.hidden_dim
        kv_dim, kv_mul, hs = self.kv_dim, self.kv_mul, self.head_size
        seq_len, n_heads = self.seq_len, self.n_heads
        data = self.data
        kc, vc = self.key_cache, self.value_cache
        exp = math.exp
        inf = math.inf

        e = self.o_emb + token * dim
        x = data[e:e + dim].tolist()

        # RoPE cos/sin for this position (float32 like cosf/sinf of float32 arg)
        rope = []
        for freq in self.rope_freq:
            val = f32(pos * freq)
            rope.append((f32(math.cos(val)), f32(math.sin(val))))

        npos = pos + 1
        scale = self.inv_sqrt_hs
        vlen = (npos - 1) * kv_dim + 1   # span of one strided value column over t = 0..pos

        for l in range(self.n_layers):
            xb = self.rmsnorm(x, self.o_rms_att + l * dim)

            q = self.matmul(xb, self.o_wq + l * dim * dim, dim, dim)
            k = self.matmul(xb, self.o_wk + l * dim * kv_dim, dim, kv_dim)
            v = self.matmul(xb, self.o_wv + l * dim * kv_dim, dim, kv_dim)

            # RoPE
            for i in range(0, dim, 2):
                fcr, fci = rope[i >> 1]
                v0 = q[i]; v1 = q[i + 1]
                q[i] = v0 * fcr - v1 * fci
                q[i + 1] = v0 * fci + v1 * fcr
                if i < kv_dim:
                    v0 = k[i]; v1 = k[i + 1]
                    k[i] = v0 * fcr - v1 * fci
                    k[i + 1] = v0 * fci + v1 * fcr
            q = rf32(q)

            # store k, v in cache at this position
            loff = l * seq_len * kv_dim
            co = loff + pos * kv_dim
            kc[co:co + kv_dim] = array('f', k)
            vc[co:co + kv_dim] = array('f', v)

            # multihead attention
            xb = [0.0] * dim
            for h in range(n_heads):
                qo = h * hs
                ko = loff + (h // kv_mul) * hs
                # scores: row t of the key cache (stride kv_dim) . q_h
                att = rf32(dots(kc, ko, kv_dim, 1, hs, npos, q[qo:qo + hs]))
                att = rf32([s / scale for s in att])
                # softmax over 0..pos
                mx = att[0]
                for s in att:
                    if s > mx:
                        mx = s
                att = rf32([exp(s) for s in rf32([s - mx for s in att])])
                tot = 0.0
                for s in att:
                    tot += s
                tot = f32(tot)
                att = rf32([s / tot for s in att])
                # weighted sum of values: out[i] = sum_t att[t] * v_t[i], t ascending.
                # Column i of the value cache is a strided slice (step kv_dim).
                xb[qo:qo + hs] = rf32(dots(vc, ko, 1, kv_dim, npos, hs, att, vlen))

            xb2 = self.matmul(xb, self.o_wo + l * dim * dim, dim, dim)
            x = rf32([a + b for a, b in zip(x, xb2)])

            xb = self.rmsnorm(x, self.o_rms_ffn + l * dim)
            hb = self.matmul(xb, self.o_w1 + l * dim * hd, dim, hd)
            hb2 = self.matmul(xb, self.o_w3 + l * dim * hd, dim, hd)
            # SwiGLU, float32 at every step: expf(-val) (inf on overflow), 1+e, 1/(...),
            # val*sig, *hb2
            ex = rf32([exp(-a) if a > -709.0 else inf for a in hb])
            sig = rf32([1.0 / a for a in rf32([1.0 + a for a in ex])])
            sv = rf32([a * b for a, b in zip(hb, sig)])
            hb = rf32([a * b for a, b in zip(sv, hb2)])
            xb = self.matmul(hb, self.o_w2 + l * dim * hd, hd, dim)
            x = rf32([a + b for a, b in zip(x, xb)])

        x = self.rmsnorm(x, self.o_rms_final)
        return array('f', dots(data, self.o_wcls, dim, 1, dim, self.vocab_size, x))


def _dots_indexed(src, off, rstride, estride, n, d, x, span=None):
    """Same contract and summation order as _dots_zip, written as a plain indexed loop.

    Used on PyPy, whose JIT compiles this form well (~4 ns/MAC vs ~48 ns/MAC for the
    zip-over-slices form); on CPython it is the slower of the two (~39 vs ~21 ns/MAC).
    """
    out = [0.0] * d
    d4 = d - (d & 3)
    for i in range(0, d4, 4):
        b0 = off + i * rstride
        b1 = b0 + rstride
        b2 = b1 + rstride
        b3 = b2 + rstride
        v0 = v1 = v2 = v3 = 0.0
        for j in range(n):
            c = x[j]
            e = j * estride
            v0 += src[b0 + e] * c
            v1 += src[b1 + e] * c
            v2 += src[b2 + e] * c
            v3 += src[b3 + e] * c
        out[i] = v0
        out[i + 1] = v1
        out[i + 2] = v2
        out[i + 3] = v3
    for i in range(d4, d):
        b0 = off + i * rstride
        v0 = 0.0
        for j in range(n):
            v0 += src[b0 + j * estride] * x[j]
        out[i] = v0
    return out


def _dots_zip(src, off, rstride, estride, n, d, x, span=None):
    """Return [sum_j src[off + i*rstride + j*estride] * x[j] for i in range(d)].

    Each sum is accumulated left to right (j ascending), exactly as the C loops do;
    four output rows share one pass over x to amortise interpreter loop overhead.
    """
    if span is None:
        span = (n - 1) * estride + 1
    out = [0.0] * d
    d4 = d - (d & 3)
    r1, r2, r3 = rstride, 2 * rstride, 3 * rstride
    for i in range(0, d4, 4):
        b = off + i * rstride
        v0 = v1 = v2 = v3 = 0.0
        for c, a0, a1, a2, a3 in zip(x,
                                     src[b:b + span:estride],
                                     src[b + r1:b + r1 + span:estride],
                                     src[b + r2:b + r2 + span:estride],
                                     src[b + r3:b + r3 + span:estride]):
            v0 += a0 * c
            v1 += a1 * c
            v2 += a2 * c
            v3 += a3 * c
        out[i] = v0
        out[i + 1] = v1
        out[i + 2] = v2
        out[i + 3] = v3
    for i in range(d4, d):
        b = off + i * rstride
        v0 = 0.0
        for c, a0 in zip(x, src[b:b + span:estride]):
            v0 += a0 * c
        out[i] = v0
    return out


# Both kernels are scalar loops with identical arithmetic and order (bit-identical
# results); pick the one the running interpreter executes fastest.
dots = _dots_indexed if sys.implementation.name == 'pypy' else _dots_zip


def argmax(logits):
    best_i = 0
    best = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best:
            best = logits[i]
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Tokenizer

class Tokenizer:
    def __init__(self, path, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = []
        self.scores = []
        with open(path, 'rb') as f:
            (self.max_token_length,) = struct.unpack('<i', f.read(4))
            for _ in range(vocab_size):
                score, n = struct.unpack('<fi', f.read(8))
                s = f.read(n)
                # C stores as NUL-terminated string
                z = s.find(b'\0')
                if z >= 0:
                    s = s[:z]
                self.scores.append(score)
                self.vocab.append(s)
        # lookup table (exact string match, equivalent to bsearch on sorted vocab)
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            if s not in self.lookup:
                self.lookup[s] = i

    def str_lookup(self, s):
        return self.lookup.get(s, -1)

    def encode(self, text, bos=True, eos=False):
        z = text.find(b'\0')
        if z >= 0:
            text = text[:z]
        tokens = []
        if bos:
            tokens.append(1)
        if len(text) > 0:
            tokens.append(self.str_lookup(b' '))
        buf = bytearray()
        n = len(text)
        for idx in range(n):
            c = text[idx]
            if (c & 0xC0) != 0x80:
                buf = bytearray()
            buf.append(c)
            nxt = text[idx + 1] if idx + 1 < n else 0
            if (nxt & 0xC0) == 0x80 and len(buf) < 4:
                continue
            tid = self.str_lookup(bytes(buf))
            if tid != -1:
                tokens.append(tid)
            else:
                for b in buf:
                    tokens.append(b + 3)
            buf = bytearray()

        vocab, scores = self.vocab, self.scores
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                tid = self.str_lookup(vocab[tokens[i]] + vocab[tokens[i + 1]])
                if tid != -1 and scores[tid] > best_score:
                    best_score = scores[tid]
                    best_id = tid
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]
        if eos:
            tokens.append(2)
        return tokens

    def decode(self, prev_token, token):
        piece = self.vocab[token]
        if prev_token == 1 and piece[:1] == b' ':
            piece = piece[1:]
        # emulate sscanf(piece, "<0x%02hhX>", &byte_val) == 1
        if piece.startswith(b'<0x'):
            j = 3
            while j < len(piece) and piece[j] in b' \t\n\v\f\r':
                j += 1
            hexd = b''
            while len(hexd) < 2 and j < len(piece) and piece[j] in b'0123456789abcdefABCDEF':
                hexd += piece[j:j + 1]
                j += 1
            if hexd:
                piece = bytes([int(hexd, 16)])
        # the C piece is a NUL-terminated string: the <0x00> byte piece decodes to ""
        z = piece.find(b'\0')
        if z >= 0:
            piece = piece[:z]
        return piece


# ---------------------------------------------------------------------------
# Protocol

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('usage: run.py <checkpoint.bin> <tokenizer.bin>\n')
        sys.exit(1)
    model = Transformer(sys.argv[1])
    tok = Tokenizer(sys.argv[2], model.vocab_size)
    out = sys.stdout

    def say(line):
        out.write(line + '\n')
        out.flush()

    say('READY')
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.readline()
        if not raw:
            break
        if raw.endswith(b'\n'):
            raw = raw[:-1]
        if raw.endswith(b'\r'):
            raw = raw[:-1]
        cmd, _, rest = raw.partition(b' ')
        cmd = cmd.decode('ascii', 'replace')

        if cmd == 'QUIT':
            break

        elif cmd == 'ENCODE':
            ids = tok.encode(rest)
            say('IDS ' + ' '.join(map(str, ids)))

        elif cmd == 'PIECE':
            a, b = rest.split()
            say('HEX ' + tok.decode(int(a), int(b)).hex())

        elif cmd == 'GENERATE':
            mn, _, text = rest.partition(b' ')
            max_new = int(mn)
            prompt = tok.encode(text)
            model.reset()
            count = 0
            reason = 'max'
            if max_new > 0:
                token = prompt[0]
                pos = 0
                while True:
                    if pos >= model.seq_len:
                        reason = 'context'
                        break
                    logits = model.forward(token, pos)
                    pos += 1
                    if pos < len(prompt):
                        token = prompt[pos]
                        continue
                    nxt = argmax(logits)
                    if nxt == 1:
                        reason = 'stop'
                        break
                    say('T %d' % nxt)
                    count += 1
                    token = nxt
                    if count >= max_new:
                        reason = 'max'
                        break
            say('DONE %d %s' % (count, reason))

        elif cmd == 'FIXED':
            ids = [int(t) for t in rest.split()]
            model.reset()
            am = []
            for pos, t in enumerate(ids[:model.seq_len]):
                am.append(argmax(model.forward(t, pos)))
            say('DONE %d %s' % (len(am), ' '.join(map(str, am))))

        elif cmd == 'LOGITS':
            parts = rest.split()
            path = parts[0].decode()
            ids = [int(t) for t in parts[1:]]
            model.reset()
            n = 0
            with open(path, 'wb') as f:
                for pos, t in enumerate(ids[:model.seq_len]):
                    lg = model.forward(t, pos)
                    if sys.byteorder != 'little':
                        lg.byteswap()
                    f.write(lg.tobytes())
                    n += 1
            say('DONE %d' % n)

        else:
            say('ERROR unknown command')


if __name__ == '__main__':
    main()
