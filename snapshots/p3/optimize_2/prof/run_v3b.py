#!/usr/bin/env python3
"""P3 port of llama2.c (reference/run.c): NumPy float32 arrays, native BLAS via np.dot / np.matmul.

Usage: python3 run.py <checkpoint.bin> <tokenizer.bin>, then the line protocol from INTERFACE.md on stdin.
"""
import heapq
import math
import struct
import sys

import numpy as np

F32 = np.float32


# ----------------------------------------------------------------------------
# Column-blocked matrix-vector product
#
# run.c's matmul(xout, x, W, n, d) computes xout[i] = sum_j W[i, j] * x[j] for a row-major (d, n) W.
# Accelerate's sgemv streams memory fastest when rows are short (measured 53-58 GB/s for 128-256
# floats vs ~45 GB/s for row-wise dots over 768-2048 floats), so each W is stored as n/b contiguous
# column blocks W_k = W[:, k*b:(k+1)*b] of shape (d, b). The product is then one batched float32
# gemv per matrix (np.matmul over the blocks) and a float32 sum of the n/b partial results:
# the same multiply-adds over the same float32 weights, grouped in blocks of b.

def block_size(n):
    for b in (256, 128, 192, 384, 96, 64, 32):
        if n % b == 0 and n > b:
            return b
    return n


class Reader:
    """Reads float32 arrays from a checkpoint file sequentially."""
    CHUNK = 1 << 20  # floats per scratch read (4 MB)

    def __init__(self, f):
        self.f = f
        self.scratch = np.empty(self.CHUNK, F32)

    def fill(self, a):
        mv = memoryview(a).cast('B')
        got = 0
        while got < len(mv):
            n = self.f.readinto(mv[got:])
            if not n:
                raise EOFError('checkpoint is truncated')
            got += n

    def plain(self, *shape):
        a = np.empty(shape, F32)
        self.fill(a)
        return a

    def blocked(self, lead, d, n, b):
        """lead row-major (d, n) matrices -> contiguous (lead, n/b, d, b) column blocks."""
        nb = n // b
        out = np.empty((lead, nb, d, b), F32)
        rows = max(1, self.CHUNK // n)
        for l in range(lead):
            for r in range(0, d, rows):
                k = min(rows, d - r)
                buf = self.scratch[:k * n]
                self.fill(buf)
                out[l, :, r:r + k, :] = buf.reshape(k, nb, b).transpose(1, 0, 2)
        return out


# ----------------------------------------------------------------------------
# Transformer model

class Transformer:
    def __init__(self, path):
        with open(path, 'rb') as f:
            hdr = f.read(28)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('<7i', hdr)
        shared = vocab_size > 0  # negative vocab size signals unshared classifier weights
        vocab_size = abs(vocab_size)
        self.dim, self.hidden_dim, self.n_layers = dim, hidden_dim, n_layers
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.vocab_size, self.seq_len = vocab_size, seq_len
        hs = dim // n_heads
        self.head_size = hs
        kv_dim = (dim * n_kv_heads) // n_heads
        self.kv_dim = kv_dim
        self.kv_mul = n_heads // n_kv_heads

        # Stream the checkpoint once, in file order, through a small scratch buffer, copying every
        # matrix into its column-blocked layout (see blocked_matvec notes above). Vectors are read as is.
        L = n_layers
        bd, bh = block_size(dim), block_size(hidden_dim)
        self.bd, self.bh = bd, bh
        with open(path, 'rb', buffering=0) as f:
            f.seek(28)
            rd = Reader(f)
            emb_blocked = rd.blocked(1, vocab_size, dim, bd)[0] if shared else None
            emb = None if shared else rd.plain(vocab_size, dim)
            self.rms_att = rd.plain(L, dim)
            wq = rd.blocked(L, n_heads * hs, dim, bd)
            wk = rd.blocked(L, kv_dim, dim, bd)
            wv = rd.blocked(L, kv_dim, dim, bd)
            wo = rd.blocked(L, dim, n_heads * hs, bd)
            self.rms_ffn = rd.plain(L, dim)
            w1 = rd.blocked(L, hidden_dim, dim, bd)
            w2 = rd.blocked(L, dim, hidden_dim, bh)
            w3 = rd.blocked(L, hidden_dim, dim, bd)
            self.rms_final = rd.plain(dim)
            f.seek(2 * (seq_len * hs // 2) * 4, 1)  # skip legacy freq_cis_real / freq_cis_imag
            self.wcls = emb_blocked if shared else rd.blocked(1, vocab_size, dim, bd)[0]
        # Token embedding rows: with shared weights they are read back out of the blocked classifier.
        self.tok_emb = emb

        # RoPE rotation table, float32, same formula as the reference:
        # freq = 1 / powf(10000, head_dim / head_size); val = pos * freq; cosf / sinf.
        # Stored as complex64 (cos + i*sin) so that rotating a pair (v0, v1) is one complex multiply:
        # (v0 + i*v1) * (c + i*s) = (v0*c - v1*s) + i*(v0*s + v1*c), the reference formula.
        head_dim = np.arange(0, hs, 2, dtype=np.int32).astype(F32)
        freq = (F32(1.0) / np.power(F32(10000.0), head_dim / F32(hs))).astype(F32)
        pos = np.arange(seq_len, dtype=np.int32).astype(F32)
        val = (pos[:, None] * freq[None, :]).astype(F32)
        rope = np.empty((seq_len, hs // 2, 2), dtype=F32)
        rope[:, :, 0] = np.cos(val)
        rope[:, :, 1] = np.sin(val)
        self.rope = rope.view(np.complex64)[:, :, 0]  # (seq_len, hs/2) complex64

        self.layers = [(self.rms_att[l], wq[l], wk[l], wv[l], wo[l], self.rms_ffn[l], w1[l], w3[l], w2[l])
                       for l in range(L)]

        # KV cache. Keys: (layer, kv_head, seq_len, head_size), so scores are one batched q.K^T.
        # Values are kept in two layouts (written together, identical contents) for the weighted sum
        # out[d] = sum_t att[t] * v[t, d], which both evaluate as a float32 sum over t in order:
        #   value_cache   (layer, kv_head, seq_len, head_size): einsum, fastest for short contexts;
        #   value_cache_t (layer, kv_head, head_size, seq_len): row dots over contiguous t (vecdot),
        #   fastest once rows are long (measured crossover ~750 positions, see NOTES.md).
        self.key_cache = np.zeros((L, n_kv_heads, seq_len, hs), dtype=F32)
        self.value_cache = np.zeros((L, n_kv_heads, seq_len, hs), dtype=F32)
        self.value_cache_t = np.zeros((L, n_kv_heads, hs, seq_len), dtype=F32)
        self.wsum_switch = 768
        self.used = 0  # number of cache positions written since the last reset

        # activation buffers
        self.x = np.zeros(dim, F32)
        self.xb = np.zeros(dim, F32)
        self.xb2 = np.zeros(dim, F32)
        self.hb = np.zeros(hidden_dim, F32)
        self.hb2 = np.zeros(hidden_dim, F32)
        self.q = np.zeros(n_heads * hs, F32)
        self.qc = self.q.view(np.complex64).reshape(n_heads, hs // 2)
        self.qg = self.q.reshape(n_kv_heads, self.kv_mul, hs)  # query heads grouped by their kv head
        self.logits = np.zeros(vocab_size, F32)
        # blocked views of the matvec inputs, and partial-sum buffers (one row per column block)
        self.xb_s = self.xb.reshape(dim // bd, bd, 1)
        self.hb_s = self.hb.reshape(hidden_dim // bh, bh, 1)
        self.x_s = self.x.reshape(dim // bd, bd, 1)
        self.p_q = np.zeros((dim // bd, n_heads * hs, 1), F32)
        self.p_kv = np.zeros((dim // bd, kv_dim, 1), F32)
        self.p_kv_h = self.p_kv.reshape(dim // bd, n_kv_heads, hs)
        self.p_o = np.zeros((dim // bd, dim, 1), F32)
        self.p_h = np.zeros((dim // bd, hidden_dim, 1), F32)
        self.p_d = np.zeros((hidden_dim // bh, dim, 1), F32)
        self.p_cls = np.zeros((dim // bd, vocab_size, 1), F32)
        self.sqrt_hs = F32(math.sqrt(hs))  # sqrtf(head_size); exact for an integer head size
        self.fdim = F32(dim)
        self.eps = F32(1e-5)
        self.one = F32(1.0)

    def reset(self):
        """Clean KV cache: zero every position written by the previous command."""
        if self.used:
            self.key_cache[:, :, :self.used, :] = 0
            self.value_cache[:, :, :self.used, :] = 0
            self.value_cache_t[:, :, :, :self.used] = 0
            self.used = 0

    def forward(self, token, pos):
        matmul, add_reduce, multiply, exp, vecdot = np.matmul, np.add.reduce, np.multiply, np.exp, np.vecdot
        one, eps, fdim, sqrt_hs = self.one, self.eps, self.fdim, self.sqrt_hs
        x, xb, xb2, hb, hb2 = self.x, self.xb, self.xb2, self.hb, self.hb2
        xb_s, hb_s = self.xb_s, self.hb_s
        p_q, p_kv, p_kv_h, p_o, p_h, p_d = self.p_q, self.p_kv, self.p_kv_h, self.p_o, self.p_h, self.p_d
        q, qc, qg = self.q, self.qc, self.qg
        rot = self.rope[pos]
        wsum_switch = self.wsum_switch
        T = pos + 1
        if T > self.used:
            self.used = T

        if self.tok_emb is None:
            x.reshape(self.wcls.shape[0], -1)[:] = self.wcls[:, token, :]
        else:
            x[:] = self.tok_emb[token]

        # Every W @ v below is matvec(): out[i] = sum over column blocks k of (W_k @ v_k)[i].
        for l, (rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2) in enumerate(self.layers):
            # attention rmsnorm: weight * (ss * x), ss = 1/sqrt(sum(x^2)/dim + 1e-5)
            ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
            multiply(x, ss, out=xb)
            multiply(rms_att, xb, out=xb)

            kc = self.key_cache[l]
            vc = self.value_cache[l]
            vct = self.value_cache_t[l]
            kslot = kc[:, pos, :]
            vslot = vc[:, pos, :]
            matmul(wq, xb_s, out=p_q)
            add_reduce(p_q[:, :, 0], axis=0, out=q)
            matmul(wk, xb_s, out=p_kv)
            add_reduce(p_kv_h, axis=0, out=kslot)          # k written straight into the cache slot
            matmul(wv, xb_s, out=p_kv)
            add_reduce(p_kv_h, axis=0, out=vslot)          # v likewise
            vct[:, :, pos] = vslot

            # RoPE on q and on the cached k
            multiply(qc, rot, out=qc)
            kcx = kslot.view(np.complex64)
            multiply(kcx, rot, out=kcx)

            # multihead attention over positions 0..pos
            att = matmul(qg, kc[:, :T, :].transpose(0, 2, 1))  # (kv_heads, kv_mul, T)
            att /= sqrt_hs
            att -= att.max(axis=-1, keepdims=True)
            exp(att, out=att)
            att /= att.sum(axis=-1, keepdims=True, dtype=F32)
            if T < wsum_switch:
                np.einsum('hmt,htd->hmd', att, vc[:, :T, :], out=xb.reshape(qg.shape))
            else:
                vecdot(vct[:, None, :, :T], att[:, :, None, :], out=xb.reshape(qg.shape))

            matmul(wo, xb_s, out=p_o)
            add_reduce(p_o[:, :, 0], axis=0, out=xb2)
            x += xb2

            # ffn rmsnorm
            ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
            multiply(x, ss, out=xb)
            multiply(rms_ffn, xb, out=xb)
            matmul(w1, xb_s, out=p_h)
            add_reduce(p_h[:, :, 0], axis=0, out=hb)
            matmul(w3, xb_s, out=p_h)
            add_reduce(p_h[:, :, 0], axis=0, out=hb2)
            # SwiGLU: hb = hb * (1 / (1 + exp(-hb))) * hb2
            sig = exp(np.negative(hb))
            sig += one
            np.divide(one, sig, out=sig)
            hb *= sig
            hb *= hb2
            matmul(w2, hb_s, out=p_d)
            add_reduce(p_d[:, :, 0], axis=0, out=xb)
            x += xb

        ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
        multiply(x, ss, out=x)
        multiply(self.rms_final, x, out=x)
        matmul(self.wcls, self.x_s, out=self.p_cls)
        add_reduce(self.p_cls[:, :, 0], axis=0, out=self.logits)
        return self.logits


# ----------------------------------------------------------------------------
# BPE tokenizer

class Tokenizer:
    def __init__(self, path, vocab_size):
        with open(path, 'rb') as f:
            data = f.read()
        self.max_token_length = struct.unpack_from('<i', data, 0)[0]
        off = 4
        self.vocab = []
        self.scores = []
        for _ in range(vocab_size):
            score, n = struct.unpack_from('<fi', data, off)
            off += 8
            s = data[off:off + n]
            off += n
            nul = s.find(b'\0')  # vocab entries are C strings
            if nul >= 0:
                s = s[:nul]
            self.vocab.append(s)
            self.scores.append(score)  # exact float32 value, compared as in C
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            self.lookup.setdefault(s, i)

    def decode(self, prev_token, token):
        piece = self.vocab[token]
        if prev_token == 1 and piece[:1] == b' ':
            piece = piece[1:]
        # sscanf(piece, "<0x%02hhX>", &byte_val) == 1
        if piece[:3] == b'<0x':
            h = piece[3:5]
            n = 0
            while n < len(h) and h[n:n + 1] in b'0123456789abcdefABCDEF' and h[n:n + 1] != b'':
                n += 1
            if n > 0:
                b = int(h[:n], 16)
                piece = bytes([b]) if b != 0 else b''  # byte_pieces entry is a C string
        return piece

    def encode(self, text):
        lookup = self.lookup
        tokens = [1]
        if text:
            tokens.append(lookup.get(b' ', -1))
        n = len(text)
        i = 0
        buf = bytearray()
        while i < n:
            c = text[i]
            if (c & 0xC0) != 0x80:
                buf = bytearray()
            buf.append(c)
            nxt = text[i + 1] if i + 1 < n else 0
            i += 1
            if (nxt & 0xC0) == 0x80 and len(buf) < 4:
                continue
            tid = lookup.get(bytes(buf), -1)
            if tid != -1:
                tokens.append(tid)
            else:
                tokens.extend(b + 3 for b in buf)
            buf = bytearray()

        return self._merge(tokens)

    # float32(-1e10), the initial best_score in run.c; only merges scoring strictly above it happen
    _NO_SCORE = float(np.float32(-1e10))

    def _merge(self, tokens):
        """Greedy BPE merges, identical in result to run.c's loop.

        run.c repeatedly scans all adjacent pairs and merges the one with the highest score, the
        leftmost one on ties (strict '>'). Here the candidate pairs sit in a heap keyed by
        (-score, original position of the left token); position order never changes under merging,
        so the heap top among still-valid entries is exactly the pair run.c would pick. Entries are
        validated lazily (both tokens alive, still adjacent, unchanged ids)."""
        n = len(tokens)
        if n < 2:
            return tokens
        vocab, scores, lookup, floor = self.vocab, self.scores, self.lookup, self._NO_SCORE
        tok = list(tokens)
        nxt = list(range(1, n + 1))      # nxt[i] == n means no right neighbour
        prv = list(range(-1, n - 1))
        alive = [True] * n
        def cand(i, j):
            tid = lookup.get(vocab[tok[i]] + vocab[tok[j]], -1)
            if tid != -1 and scores[tid] > floor:
                return (-scores[tid], i, j, tok[i], tok[j], tid)
            return None

        heap = [c for c in (cand(i, i + 1) for i in range(n - 1)) if c is not None]
        heapq.heapify(heap)
        hpush, hpop = heapq.heappush, heapq.heappop
        while heap:
            _, i, j, ti, tj, tid = hpop(heap)
            if not (alive[i] and alive[j] and nxt[i] == j and tok[i] == ti and tok[j] == tj):
                continue  # stale candidate
            tok[i] = tid
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k < n:
                prv[k] = i
            p = prv[i]
            if p >= 0:
                c = cand(p, i)
                if c is not None:
                    hpush(heap, c)
            if k < n:
                c = cand(i, k)
                if c is not None:
                    hpush(heap, c)
        out = []
        i = 0
        while i < n:
            out.append(tok[i])
            i = nxt[i]
        return out


# ----------------------------------------------------------------------------
# command loop

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('usage: run.py <checkpoint.bin> <tokenizer.bin>\n')
        sys.exit(1)
    model = Transformer(sys.argv[1])
    tok = Tokenizer(sys.argv[2], model.vocab_size)
    out = sys.stdout

    def say(s):
        out.write(s + '\n')
        out.flush()

    say('READY')
    inp = sys.stdin.buffer
    while True:
        raw = inp.readline()
        if not raw:
            break
        if raw.endswith(b'\n'):
            raw = raw[:-1]
        cmd, _, rest = raw.partition(b' ')
        cmd = cmd.decode('ascii', 'replace')

        if cmd == 'QUIT':
            break
        elif cmd == 'ENCODE':
            say('IDS ' + ' '.join(map(str, tok.encode(rest))))
        elif cmd == 'PIECE':
            a, b = rest.split()[:2]
            say('HEX ' + tok.decode(int(a), int(b)).hex())
        elif cmd == 'GENERATE':
            n_str, _, text = rest.partition(b' ')
            max_new = int(n_str)
            prompt = tok.encode(text)
            model.reset()
            seq_len = model.seq_len
            count = 0
            reason = 'max'
            token = prompt[0]
            pos = 0
            if max_new <= 0:
                reason = 'max'
            else:
                while True:
                    if pos >= seq_len:
                        reason = 'context'
                        break
                    logits = model.forward(token, pos)
                    if pos < len(prompt) - 1:
                        nxt = prompt[pos + 1]
                        pos += 1
                        token = nxt
                        continue
                    nxt = int(np.argmax(logits))
                    pos += 1
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
        elif cmd == 'FIXED' or cmd == 'LOGITS':
            parts = rest.split()
            if cmd == 'LOGITS':
                path, parts = parts[0], parts[1:]
            ids = [int(t) for t in parts]
            model.reset()
            n = len(ids)
            if cmd == 'LOGITS':
                allv = np.empty((n, model.vocab_size), dtype=F32)
                for p, t in enumerate(ids):
                    allv[p] = model.forward(t, p)
                allv.astype('<f4', copy=False).tofile(path.decode('utf-8', 'surrogateescape'))
                say('DONE %d' % n)
            else:
                am = []
                for p, t in enumerate(ids):
                    am.append(int(np.argmax(model.forward(t, p))))
                say(' '.join(['DONE', str(n)] + [str(a) for a in am]))
        else:
            sys.stderr.write('unknown command: %r\n' % raw)
    out.flush()
    sys.exit(0)


if __name__ == '__main__':
    main()
