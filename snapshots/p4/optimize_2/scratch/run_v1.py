#!/usr/bin/env python3
"""P4 port of reference/run.c (llama2.c): Numba nopython kernels on float32 arrays.

Usage: python3 run.py <checkpoint.bin> <tokenizer.bin>, then the line protocol in INTERFACE.md.

Compute: every arithmetic loop of the forward pass is an @njit (nopython) function working
on float32 NumPy arrays. No BLAS (no np.dot / @), no fastmath, no parallel/threads.
NumPy is used only to hold arrays (load, reshape/transpose copies, writing logits).
"""
import math
import struct
import sys

import numpy as np
from numba import njit

F32 = np.float32

# ----------------------------------------------------------------------------
# compiled kernels


@njit(cache=True, boundscheck=False)
def matmul_t(out, x, wt):
    """out[i] = sum_j wt[j, i] * x[j], j ascending (same per-output order as run.c matmul).

    wt is the transposed weight W^T with shape (n, d). The inner loop runs over the outputs,
    which vectorizes without reassociating any sum. Two input rows per pass to cut the
    loads/stores of out; the two adds are still applied one after another, in j order.
    Used for the per-layer matrices (d <= hidden_dim).
    """
    n = wt.shape[0]
    d = wt.shape[1]
    for i in range(d):
        out[i] = F32(0.0)
    j = 0
    while j + 2 <= n:
        x0 = x[j]
        x1 = x[j + 1]
        w0 = wt[j]
        w1 = wt[j + 1]
        for i in range(d):
            v = out[i]
            v += w0[i] * x0
            v += w1[i] * x1
            out[i] = v
        j += 2
    while j < n:
        xj = x[j]
        wj = wt[j]
        for i in range(d):
            out[i] += wj[i] * xj
        j += 1


@njit(cache=True, boundscheck=False)
def matmul_t4(out, x, wt):
    """Same as matmul_t (identical results), four input rows per pass. Faster for the
    wide classifier (d = vocab), where out does not fit in L1."""
    n = wt.shape[0]
    d = wt.shape[1]
    for i in range(d):
        out[i] = F32(0.0)
    j = 0
    while j + 4 <= n:
        x0 = x[j]
        x1 = x[j + 1]
        x2 = x[j + 2]
        x3 = x[j + 3]
        w0 = wt[j]
        w1 = wt[j + 1]
        w2 = wt[j + 2]
        w3 = wt[j + 3]
        for i in range(d):
            v = out[i]
            v += w0[i] * x0
            v += w1[i] * x1
            v += w2[i] * x2
            v += w3[i] * x3
            out[i] = v
        j += 4
    while j < n:
        xj = x[j]
        wj = wt[j]
        for i in range(d):
            out[i] += wj[i] * xj
        j += 1


@njit(cache=True, boundscheck=False)
def rmsnorm(o, x, weight):
    size = x.shape[0]
    ss = F32(0.0)
    for j in range(size):
        ss += x[j] * x[j]
    ss = ss / F32(size)
    ss += F32(1e-5)
    ss = F32(1.0) / math.sqrt(ss)
    for j in range(size):
        o[j] = weight[j] * (ss * x[j])


@njit(cache=True, boundscheck=False)
def softmax(x, size):
    max_val = x[0]
    for i in range(1, size):
        if x[i] > max_val:
            max_val = x[i]
    s = F32(0.0)
    for i in range(size):
        x[i] = math.exp(x[i] - max_val)
        s += x[i]
    for i in range(size):
        x[i] = x[i] / s


@njit(cache=True, boundscheck=False)
def argmax(v):
    best = 0
    bv = v[0]
    for i in range(1, v.shape[0]):
        if v[i] > bv:
            bv = v[i]
            best = i
    return best


@njit(cache=True, boundscheck=False)
def forward(token, pos, emb, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, wcls,
            key_cache, value_cache, x, xb, xb2, hb, hb2, q, k, v, att, logits, n_heads, n_kv_heads):
    """One position of the transformer, same operations and order as run.c forward().

    key_cache: (layer, n_kv_heads, head_size, seq_len)  -- keys stored transposed per head
    value_cache: (layer, seq_len, kv_dim)
    Weights w*: (layer, in, out) i.e. transposed relative to the checkpoint.
    """
    dim = x.shape[0]
    hidden_dim = hb.shape[0]
    n_layers = wq.shape[0]
    head_size = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads
    kv_mul = n_heads // n_kv_heads
    seq_len = value_cache.shape[1]
    hs_f = F32(head_size)
    sqrt_hs = math.sqrt(hs_f)

    for i in range(dim):
        x[i] = emb[token, i]

    for l in range(n_layers):
        rmsnorm(xb, x, rms_att[l])

        matmul_t(q, xb, wq[l])
        matmul_t(k, xb, wk[l])
        matmul_t(v, xb, wv[l])

        # RoPE
        for i in range(0, dim, 2):
            head_dim = i % head_size
            freq = F32(1.0) / math.pow(F32(10000.0), F32(head_dim) / hs_f)
            val = F32(pos) * freq
            fcr = math.cos(val)
            fci = math.sin(val)
            v0 = q[i]
            v1 = q[i + 1]
            q[i] = v0 * fcr - v1 * fci
            q[i + 1] = v0 * fci + v1 * fcr
            if i < kv_dim:
                v0 = k[i]
                v1 = k[i + 1]
                k[i] = v0 * fcr - v1 * fci
                k[i + 1] = v0 * fci + v1 * fcr

        # store k, v at this position in the cache
        kc = key_cache[l]
        vc = value_cache[l]
        for hk in range(n_kv_heads):
            for i in range(head_size):
                kc[hk, i, pos] = k[hk * head_size + i]
        for i in range(kv_dim):
            vc[pos, i] = v[i]

        # attention, per head
        n_t = pos + 1
        for h in range(n_heads):
            hk = h // kv_mul
            kh = kc[hk]
            a = att[h]
            for t in range(n_t):
                a[t] = F32(0.0)
            qo = h * head_size
            for i in range(head_size):
                qi = q[qo + i]
                kr = kh[i]
                for t in range(n_t):
                    a[t] += qi * kr[t]
            for t in range(n_t):
                a[t] = a[t] / sqrt_hs
            softmax(a, n_t)
            xo = h * head_size
            vo = hk * head_size
            for i in range(head_size):
                xb[xo + i] = F32(0.0)
            for t in range(n_t):
                at = a[t]
                vr = vc[t]
                for i in range(head_size):
                    xb[xo + i] += at * vr[vo + i]

        matmul_t(xb2, xb, wo[l])
        for i in range(dim):
            x[i] += xb2[i]

        rmsnorm(xb, x, rms_ffn[l])
        matmul_t(hb, xb, w1[l])
        matmul_t(hb2, xb, w3[l])
        for i in range(hidden_dim):
            val = hb[i]
            val *= F32(1.0) / (F32(1.0) + math.exp(-val))
            val *= hb2[i]
            hb[i] = val
        matmul_t(xb, hb, w2[l])
        for i in range(dim):
            x[i] += xb[i]

    rmsnorm(x, x, rms_final)
    matmul_t4(logits, x, wcls)


@njit(cache=True, boundscheck=False)
def transpose_into(dst, src):
    """dst (L, n, d) <- src (L, d, n) transposed, blocked for cache locality."""
    L = src.shape[0]
    d = src.shape[1]
    n = src.shape[2]
    B = 16
    for l in range(L):
        s = src[l]
        t = dst[l]
        for i0 in range(0, d, B):
            i1 = min(i0 + B, d)
            for j0 in range(0, n, B):
                j1 = min(j0 + B, n)
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        t[j, i] = s[i, j]


# ----------------------------------------------------------------------------
# model


class Model:
    def __init__(self, path):
        with open(path, 'rb') as f:
            hdr = f.read(28)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack('<7i', hdr)
        shared = vocab > 0
        vocab = abs(vocab)
        self.dim, self.hidden_dim, self.n_layers = dim, hidden_dim, n_layers
        self.n_heads, self.n_kv_heads, self.vocab, self.seq_len = n_heads, n_kv_heads, vocab, seq_len
        head_size = dim // n_heads
        kv_dim = dim * n_kv_heads // n_heads
        data = np.fromfile(path, dtype='<f4', offset=28)  # one sequential read
        off = [0]

        def take(count, shape):
            a = data[off[0]:off[0] + count].reshape(shape)
            off[0] += count
            return a

        def tposed(src):
            L, d, n = src.shape
            dst = np.empty((L, n, d), F32)
            transpose_into(dst, src)
            return dst

        L = n_layers
        emb = take(vocab * dim, (vocab, dim))
        self.emb = np.array(emb, dtype=F32)
        self.rms_att = np.array(take(L * dim, (L, dim)), dtype=F32)
        self.wq = tposed(take(L * dim * dim, (L, dim, dim)))
        self.wk = tposed(take(L * dim * kv_dim, (L, kv_dim, dim)))
        self.wv = tposed(take(L * dim * kv_dim, (L, kv_dim, dim)))
        self.wo = tposed(take(L * dim * dim, (L, dim, dim)))
        self.rms_ffn = np.array(take(L * dim, (L, dim)), dtype=F32)
        self.w1 = tposed(take(L * dim * hidden_dim, (L, hidden_dim, dim)))
        self.w2 = tposed(take(L * dim * hidden_dim, (L, dim, hidden_dim)))
        self.w3 = tposed(take(L * dim * hidden_dim, (L, hidden_dim, dim)))
        self.rms_final = np.array(take(dim, (dim,)), dtype=F32)
        off[0] += seq_len * head_size // 2  # freq_cis_real (unused)
        off[0] += seq_len * head_size // 2  # freq_cis_imag (unused)
        cls = self.emb if shared else take(vocab * dim, (vocab, dim))
        self.wcls = tposed(cls.reshape(1, vocab, dim))[0]
        del data

        self.key_cache = np.zeros((L, n_kv_heads, head_size, seq_len), F32)
        self.value_cache = np.zeros((L, seq_len, kv_dim), F32)
        self.x = np.zeros(dim, F32)
        self.xb = np.zeros(dim, F32)
        self.xb2 = np.zeros(dim, F32)
        self.hb = np.zeros(hidden_dim, F32)
        self.hb2 = np.zeros(hidden_dim, F32)
        self.q = np.zeros(dim, F32)
        self.k = np.zeros(kv_dim, F32)
        self.v = np.zeros(kv_dim, F32)
        self.att = np.zeros((n_heads, seq_len), F32)
        self.logits = np.zeros(vocab, F32)

    def reset(self):
        # clean KV cache for every command (positions restart at 0)
        self.key_cache.fill(0)
        self.value_cache.fill(0)

    def forward(self, token, pos):
        forward(token, pos, self.emb, self.rms_att, self.wq, self.wk, self.wv, self.wo,
                self.rms_ffn, self.w1, self.w2, self.w3, self.rms_final, self.wcls,
                self.key_cache, self.value_cache, self.x, self.xb, self.xb2, self.hb, self.hb2,
                self.q, self.k, self.v, self.att, self.logits, self.n_heads, self.n_kv_heads)
        return self.logits


# ----------------------------------------------------------------------------
# tokenizer (mirrors run.c build_tokenizer / encode / decode)


class Tokenizer:
    def __init__(self, path, vocab_size):
        d = open(path, 'rb').read()
        (self.max_token_length,) = struct.unpack_from('<i', d, 0)
        o = 4
        self.vocab = []
        self.scores = []
        for _ in range(vocab_size):
            s, n = struct.unpack_from('<fi', d, o)
            o += 8
            self.vocab.append(d[o:o + n])
            self.scores.append(s)
            o += n
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            # C strings: stop at NUL; bsearch over strcmp-sorted vocab finds an exact match
            self.lookup.setdefault(s.split(b'\0', 1)[0], i)

    def _pair(self, a, b):
        sid = self.lookup.get(self.vocab[a] + self.vocab[b], -1)
        return (self.scores[sid], sid) if sid != -1 else None

    def encode(self, text):
        text = text.split(b'\0', 1)[0]  # C string semantics
        tokens = [1]
        if text:
            tokens.append(self.lookup.get(b' ', -1))
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
            tid = self.lookup.get(bytes(buf), -1)
            if tid != -1:
                tokens.append(tid)
            else:
                tokens.extend(b + 3 for b in buf)
            buf = bytearray()
        # greedy merges: highest score, earliest pair on ties (strict >), as in run.c.
        # pair scores are cached and only the neighbours of a merge are recomputed;
        # the chosen merge sequence is identical to rescanning every pair each round.
        pairs = [self._pair(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        while True:
            best_score = -1e10
            best_idx = -1
            for i, p in enumerate(pairs):
                if p is not None and p[0] > best_score:
                    best_score = p[0]
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = pairs[best_idx][1]
            del tokens[best_idx + 1]
            del pairs[best_idx]
            if best_idx < len(pairs):
                pairs[best_idx] = self._pair(tokens[best_idx], tokens[best_idx + 1])
            if best_idx > 0:
                pairs[best_idx - 1] = self._pair(tokens[best_idx - 1], tokens[best_idx])
        return tokens

    def decode(self, prev, tok):
        piece = self.vocab[tok]
        if prev == 1 and piece[:1] == b' ':
            piece = piece[1:]
        if len(piece) >= 4 and piece[:3] == b'<0x':
            h = piece[3:5]
            hv = b''
            for ch in h:
                if chr(ch) in '0123456789abcdefABCDEF':
                    hv += bytes([ch])
                else:
                    break
            if hv:
                bv = int(hv, 16)
                piece = bytes([bv]) if bv != 0 else b''  # byte_pieces are C strings
        return piece.split(b'\0', 1)[0]


# ----------------------------------------------------------------------------
# protocol


def main():
    if len(sys.argv) < 3:
        sys.stderr.write('usage: run.py <checkpoint.bin> <tokenizer.bin>\n')
        sys.exit(1)
    model = Model(sys.argv[1])
    tok = Tokenizer(sys.argv[2], model.vocab)
    out = sys.stdout

    def say(s):
        out.write(s + '\n')
        out.flush()

    # warm up compiled code (results discarded; cache is reset before each command)
    model.forward(1, 0)
    argmax(model.logits)
    model.reset()
    say('READY')

    inp = sys.stdin.buffer
    while True:
        raw = inp.readline()
        if not raw:
            break
        line = raw[:-1] if raw.endswith(b'\n') else raw
        if line.endswith(b'\r'):
            line = line[:-1]
        cmd, _, rest = line.partition(b' ')
        cmd = cmd.decode('ascii', 'replace')
        if cmd == 'QUIT':
            break
        elif cmd == 'ENCODE':
            ids = tok.encode(rest)
            say('IDS ' + ' '.join(map(str, ids)))
        elif cmd == 'PIECE':
            a = rest.split()
            say('HEX ' + tok.decode(int(a[0]), int(a[1])).hex())
        elif cmd == 'GENERATE':
            n_s, _, text = rest.partition(b' ')
            max_new = int(n_s)
            ids = tok.encode(text)
            model.reset()
            seq_len = model.seq_len
            count = 0
            reason = 'max'
            if max_new > 0:
                token = ids[0]
                pos = 0
                while True:
                    if pos >= seq_len:
                        reason = 'context'
                        break
                    logits = model.forward(token, pos)
                    if pos < len(ids) - 1:
                        nxt = ids[pos + 1]
                        forced = True
                    else:
                        nxt = argmax(logits)
                        forced = False
                    pos += 1
                    if nxt == 1:
                        reason = 'stop'
                        break
                    if not forced:
                        say('T %d' % nxt)
                        count += 1
                        if count >= max_new:
                            reason = 'max'
                            break
                    token = nxt
            say('DONE %d %s' % (count, reason))
        elif cmd == 'FIXED' or cmd == 'LOGITS':
            parts = rest.split()
            path = None
            if cmd == 'LOGITS':
                path = parts[0].decode()
                parts = parts[1:]
            ids = [int(p) for p in parts][:model.seq_len]
            model.reset()
            n = len(ids)
            if path is not None:
                allv = np.empty((n, model.vocab), F32)
            am = []
            for pos, t in enumerate(ids):
                logits = model.forward(t, pos)
                if path is not None:
                    allv[pos] = logits
                else:
                    am.append(argmax(logits))
            if path is not None:
                allv.astype('<f4').tofile(path)
                say('DONE %d' % n)
            else:
                say('DONE %d %s' % (n, ' '.join(map(str, am)))) if am else say('DONE 0')
        elif cmd == '':
            continue
        else:
            say('ERR unknown command')
    out.flush()
    sys.exit(0)


if __name__ == '__main__':
    main()
