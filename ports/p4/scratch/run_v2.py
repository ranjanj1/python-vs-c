#!/usr/bin/env python3
"""P4 port of reference/run.c (llama2.c): Numba nopython kernels on float32 arrays.

Usage: python3 run.py <checkpoint.bin> <tokenizer.bin>, then the line protocol in INTERFACE.md.

Compute: every arithmetic loop of the forward pass is an @njit (nopython) function working
on float32 NumPy arrays. No BLAS (no np.dot / @), no fastmath, no parallel/threads.
NumPy is used only to hold arrays (load, reshape copies, writing logits).
"""
import math
import struct
import sys

import numpy as np
from numba import njit

F32 = np.float32
TILE = 128  # output columns per weight tile (see matvec_tiles)

# ----------------------------------------------------------------------------
# compiled kernels


@njit(cache=True, boundscheck=False)
def matvec_tile(out, x, wt):
    """out[c] = sum_j wt[j, c] * x[j], j ascending: the same per-output order as run.c matmul.

    wt is one tile of W^T, shape (n, T): column c is output row c of W. The inner loop runs
    over the T outputs, so it vectorizes without reassociating any sum (no fastmath needed).
    Four input rows per pass cut the loads/stores of out; the four adds are still applied
    one after another, in j order.
    """
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
def matvec_tiles(out, x, wt):
    """out = W @ x with W stored as contiguous column tiles wt (n_tiles, n, T).

    Each tile's T outputs stay in L1 while its (n, T) block streams through once, which
    measured ~62 GB/s vs ~54 GB/s for the untiled (n, d) layout on this M1.
    """
    T = wt.shape[2]
    for t in range(wt.shape[0]):
        matvec_tile(out[t * T:(t + 1) * T], x, wt[t])


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
def attention(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs):
    """Multi-head attention over positions 0..pos, same arithmetic order as run.c.

    kc: (n_kv_heads, n_blocks, head_size, KBLK) keys, transposed within blocks of KBLK
        positions, so each score sum_i q[i]*k[t,i] (i ascending) vectorizes over the block's t,
        and one position's key lands in one contiguous 16 KB block instead of 64 rows 4 KB
        apart (measured 1.17-1.59x faster store+attention at 40-300 positions, equal at 1000).
    vc: (seq_len, kv_dim). xb[h] = sum_t a[t]*v[t] (t ascending), vectorized over i.
    Two key rows / two timesteps per pass; adds stay in the reference order.
    """
    kv_mul = n_heads // n_kv_heads
    n_t = pos + 1
    for h in range(n_heads):
        hk = h // kv_mul
        kh = kc[hk]
        a = att[h]
        qo = h * head_size
        q0 = q[qo]
        k0 = kh[0]
        for t in range(n_t):
            a[t] = F32(0.0) + q0 * k0[t]
        i = 1
        while i + 2 <= head_size:
            qa = q[qo + i]
            qb = q[qo + i + 1]
            ka = kh[i]
            kb = kh[i + 1]
            for t in range(n_t):
                s = a[t]
                s += qa * ka[t]
                s += qb * kb[t]
                a[t] = s
            i += 2
        while i < head_size:
            qi = q[qo + i]
            kr = kh[i]
            for t in range(n_t):
                a[t] += qi * kr[t]
            i += 1
        for t in range(n_t):
            a[t] = a[t] / sqrt_hs
        softmax(a, n_t)
        vo = hk * head_size
        o = xb[qo:qo + head_size]
        for i in range(head_size):
            o[i] = F32(0.0)
        t = 0
        while t + 2 <= n_t:
            aa = a[t]
            ab = a[t + 1]
            va = vc[t, vo:vo + head_size]
            vb = vc[t + 1, vo:vo + head_size]
            for i in range(head_size):
                s = o[i]
                s += aa * va[i]
                s += ab * vb[i]
                o[i] = s
            t += 2
        while t < n_t:
            aa = a[t]
            va = vc[t, vo:vo + head_size]
            for i in range(head_size):
                o[i] += aa * va[i]
            t += 1


@njit(cache=True, boundscheck=False)
def embed(x, emb_tiles, token):
    """x = row `token` of the embedding table, read from its column-tiled copy (n_tiles, dim, T).

    With shared weights this is the classifier's tiled array, so no separate 98 MB row-major
    embedding copy is kept. The values are the same float32 numbers."""
    T = emb_tiles.shape[2]
    blk = emb_tiles[token // T]
    c = token % T
    for i in range(x.shape[0]):
        x[i] = blk[i, c]


@njit(cache=True, boundscheck=False)
def forward(token, pos, emb, rms_att, wqkv, wo, rms_ffn, w13, w2, rms_final, wcls,
            key_cache, value_cache, x, xb, xb2, qkv, h13, att, logits, rope_c, rope_s,
            n_heads, n_kv_heads, kv_dim, k_off, v_off, hidden_dim, h3_off):
    """One position of the transformer: the same operations, in the same order, as run.c forward().

    Weights are column-tiled (see matvec_tiles). wqkv holds the q, k and v tiles one after
    another (outputs land at qkv[0:], qkv[k_off:], qkv[v_off:]); w13 holds w1 then w3 tiles
    (outputs at h13[0:], h13[h3_off:]). Each output is still its own row-times-x sum.
    key_cache: (layer, n_kv_heads, n_blocks, head_size, KBLK); value_cache: (layer, seq_len, kv_dim).
    emb: tiled embedding table (n_tiles, dim, T), see embed().
    """
    dim = x.shape[0]
    n_layers = wqkv.shape[0]
    head_size = dim // n_heads
    hs_f = F32(head_size)
    sqrt_hs = math.sqrt(hs_f)

    for i in range(dim):
        x[i] = emb[token, i]

    # RoPE coefficients: the run.c expressions (powf, cosf, sinf of pos*freq). They depend only
    # on (i, pos), so they are evaluated once per position instead of once per layer; the
    # values are bit-identical.
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = F32(1.0) / math.pow(F32(10000.0), F32(head_dim) / hs_f)
        val = F32(pos) * freq
        rope_c[i // 2] = math.cos(val)
        rope_s[i // 2] = math.sin(val)

    q = qkv[:dim]
    k = qkv[k_off:k_off + kv_dim]
    v = qkv[v_off:v_off + kv_dim]
    hb = h13[:hidden_dim]
    hb2 = h13[h3_off:h3_off + hidden_dim]

    for l in range(n_layers):
        rmsnorm(xb, x, rms_att[l])
        matvec_tiles(qkv, xb, wqkv[l])

        for i in range(0, dim, 2):
            fcr = rope_c[i // 2]
            fci = rope_s[i // 2]
            v0 = q[i]
            v1 = q[i + 1]
            q[i] = v0 * fcr - v1 * fci
            q[i + 1] = v0 * fci + v1 * fcr
            if i < kv_dim:
                v0 = k[i]
                v1 = k[i + 1]
                k[i] = v0 * fcr - v1 * fci
                k[i + 1] = v0 * fci + v1 * fcr

        kc = key_cache[l]
        vc = value_cache[l]
        for hk in range(n_kv_heads):
            for i in range(head_size):
                kc[hk, i, pos] = k[hk * head_size + i]
        for i in range(kv_dim):
            vc[pos, i] = v[i]

        attention(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs)

        matvec_tiles(xb2, xb, wo[l])
        for i in range(dim):
            x[i] += xb2[i]

        rmsnorm(xb, x, rms_ffn[l])
        matvec_tiles(h13, xb, w13[l])
        for i in range(hidden_dim):
            val = hb[i]
            val *= F32(1.0) / (F32(1.0) + math.exp(-val))
            val *= hb2[i]
            hb[i] = val
        matvec_tiles(xb2, hb, w2[l])
        for i in range(dim):
            x[i] += xb2[i]

    rmsnorm(x, x, rms_final)
    matvec_tiles(logits, x, wcls)


@njit(cache=True, boundscheck=False)
def tile_into(dst, t0, src):
    """dst[l, t0 + t, j, c] = src[l, t*T + c, j]  (zero for padded columns past src's rows).

    src: (L, d, n) checkpoint matrices (row r = output r); dst: (L, n_tiles, n, T).
    Full tiles are copied in branch-free 16x16 blocks; a partial last tile, if any, is padded.
    """
    L = src.shape[0]
    d = src.shape[1]
    n = src.shape[2]
    T = dst.shape[3]
    nfull = d // T
    nb = (n // 16) * 16
    for l in range(L):
        s = src[l]
        for t in range(nfull):
            blk = dst[l, t0 + t]
            r0 = t * T
            c0 = 0
            while c0 + 16 <= T:
                for j0 in range(0, nb, 16):
                    for c in range(c0, c0 + 16):
                        sr = s[r0 + c]
                        for j in range(j0, j0 + 16):
                            blk[j, c] = sr[j]
                for j in range(nb, n):
                    for c in range(c0, c0 + 16):
                        blk[j, c] = s[r0 + c, j]
                c0 += 16
            for c in range(c0, T):
                for j in range(n):
                    blk[j, c] = s[r0 + c, j]
        if nfull * T < d:
            blk = dst[l, t0 + nfull]
            r0 = nfull * T
            for j in range(n):
                for c in range(T):
                    r = r0 + c
                    blk[j, c] = s[r, j] if r < d else F32(0.0)


# ----------------------------------------------------------------------------
# model


def _ntiles(d):
    return (d + TILE - 1) // TILE


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
        self.kv_dim = kv_dim
        # mapped read-only: weights are copied once into the tiled layout, so mapping avoids a
        # second 438 MB anonymous buffer (measured faster to READY under memory pressure)
        data = np.memmap(path, dtype='<f4', mode='r', offset=28)
        off = [0]

        def take(count, shape):
            a = data[off[0]:off[0] + count].reshape(shape)
            off[0] += count
            return a

        L = n_layers
        T = TILE
        src_emb = take(vocab * dim, (vocab, dim))
        self.emb_rows = np.array(src_emb, dtype=F32)
        self.rms_att = np.array(take(L * dim, (L, dim)), dtype=F32)
        src_q = take(L * dim * dim, (L, dim, dim))
        src_k = take(L * dim * kv_dim, (L, kv_dim, dim))
        src_v = take(L * dim * kv_dim, (L, kv_dim, dim))
        src_o = take(L * dim * dim, (L, dim, dim))
        self.rms_ffn = np.array(take(L * dim, (L, dim)), dtype=F32)
        src_1 = take(L * dim * hidden_dim, (L, hidden_dim, dim))
        src_2 = take(L * dim * hidden_dim, (L, dim, hidden_dim))
        src_3 = take(L * dim * hidden_dim, (L, hidden_dim, dim))
        self.rms_final = np.array(take(dim, (dim,)), dtype=F32)
        off[0] += seq_len * head_size // 2  # freq_cis_real (unused)
        off[0] += seq_len * head_size // 2  # freq_cis_imag (unused)
        src_cls = src_emb if shared else take(vocab * dim, (vocab, dim))

        nq, nk = _ntiles(dim), _ntiles(kv_dim)
        self.wqkv = np.empty((L, nq + 2 * nk, dim, T), F32)
        tile_into(self.wqkv, 0, src_q)
        tile_into(self.wqkv, nq, src_k)
        tile_into(self.wqkv, nq + nk, src_v)
        self.k_off, self.v_off = nq * T, (nq + nk) * T
        self.wo = np.empty((L, nq, dim, T), F32)
        tile_into(self.wo, 0, src_o)
        nh = _ntiles(hidden_dim)
        self.w13 = np.empty((L, 2 * nh, dim, T), F32)
        tile_into(self.w13, 0, src_1)
        tile_into(self.w13, nh, src_3)
        self.h3_off = nh * T
        self.w2 = np.empty((L, nq, hidden_dim, T), F32)
        tile_into(self.w2, 0, src_2)
        nv = _ntiles(vocab)
        wcls = np.empty((1, nv, dim, T), F32)
        tile_into(wcls, 0, src_cls.reshape(1, vocab, dim))
        self.wcls = wcls[0]
        if shared:
            self.emb = self.wcls  # embedding rows are read from the tiled classifier (embed())
        else:
            emb_t = np.empty((1, nv, dim, T), F32)
            tile_into(emb_t, 0, src_emb.reshape(1, vocab, dim))
            self.emb = emb_t[0]
        del data, src_emb, src_q, src_k, src_v, src_o, src_1, src_2, src_3, src_cls

        self.key_cache = np.zeros((L, n_kv_heads, head_size, seq_len), F32)
        self.value_cache = np.zeros((L, seq_len, kv_dim), F32)
        self.x = np.zeros(dim, F32)
        self.xb = np.zeros(nq * T, F32)       # padded to whole tiles (outputs of wo / w2)
        self.xb2 = np.zeros(nq * T, F32)
        self.qkv = np.zeros((nq + 2 * nk) * T, F32)
        self.h13 = np.zeros(2 * nh * T, F32)
        self.att = np.zeros((n_heads, seq_len), F32)
        self.logits_buf = np.zeros(nv * T, F32)
        self.logits = self.logits_buf[:vocab]
        self.rope_c = np.zeros(dim // 2, F32)
        self.rope_s = np.zeros(dim // 2, F32)
        self.used = 0  # positions 0..used-1 may hold KV entries; the rest are still zero

    def reset(self):
        # clean KV cache for every command (positions restart at 0)
        self.key_cache.fill(0)
        self.value_cache.fill(0)

    def forward(self, token, pos):
        forward(token, pos, self.emb_rows, self.rms_att, self.wqkv, self.wo, self.rms_ffn, self.w13,
                self.w2, self.rms_final, self.wcls, self.key_cache, self.value_cache,
                self.x, self.xb, self.xb2, self.qkv, self.h13, self.att,
                self.logits_buf, self.rope_c, self.rope_s, self.n_heads, self.n_kv_heads,
                self.kv_dim, self.k_off, self.v_off, self.hidden_dim, self.h3_off)
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
                allv = np.empty((n, model.vocab), F32)  # native on little-endian hosts
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
