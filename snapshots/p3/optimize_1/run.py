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

        data = np.fromfile(path, dtype='<f4', offset=28)  # whole checkpoint, float32
        off = 0

        def take(*shape):
            nonlocal off
            n = int(np.prod(shape))
            a = data[off:off + n].reshape(shape)
            off += n
            return a

        L = n_layers
        self.tok_emb = take(vocab_size, dim)
        self.rms_att = take(L, dim)
        self.wq = take(L, n_heads * hs, dim)        # (d, n) row-major per layer, as in matmul()
        self.wk = take(L, n_kv_heads * hs, dim)
        self.wv = take(L, n_kv_heads * hs, dim)
        self.wo = take(L, dim, n_heads * hs)
        self.rms_ffn = take(L, dim)
        self.w1 = take(L, hidden_dim, dim)
        self.w2 = take(L, dim, hidden_dim)
        self.w3 = take(L, hidden_dim, dim)
        self.rms_final = take(dim)
        off += seq_len * hs // 2  # legacy freq_cis_real
        off += seq_len * hs // 2  # legacy freq_cis_imag
        self.wcls = self.tok_emb if shared else take(vocab_size, dim)

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

        # Per-layer weight views shaped for np.vecdot (row-wise float32 dot products):
        # out[i] = sum_j W[i, j] * x[j], exactly matmul() in run.c. K/V are shaped (kv_heads, head_size, dim)
        # so their outputs land directly in the (kv_head, pos, head_size) cache slot.
        self.layers = [
            (self.rms_att[l], self.wq[l],
             self.wk[l].reshape(n_kv_heads, hs, dim), self.wv[l].reshape(n_kv_heads, hs, dim),
             self.wo[l], self.rms_ffn[l], self.w1[l], self.w3[l], self.w2[l])
            for l in range(L)]

        # KV cache laid out (layer, kv_head, seq_len, head_size) so per-head slices are contiguous.
        self.key_cache = np.zeros((L, n_kv_heads, seq_len, hs), dtype=F32)
        self.value_cache = np.zeros((L, n_kv_heads, seq_len, hs), dtype=F32)
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
        self.sqrt_hs = F32(math.sqrt(hs))  # sqrtf(head_size); exact for an integer head size
        self.fdim = F32(dim)
        self.eps = F32(1e-5)
        self.one = F32(1.0)

    def reset(self):
        """Clean KV cache: zero every position written by the previous command."""
        if self.used:
            self.key_cache[:, :, :self.used, :] = 0
            self.value_cache[:, :, :self.used, :] = 0
            self.used = 0

    def forward(self, token, pos):
        vecdot, multiply, exp = np.vecdot, np.multiply, np.exp
        dim = self.dim
        one, eps, fdim, sqrt_hs = self.one, self.eps, self.fdim, self.sqrt_hs
        x, xb, xb2, hb, hb2 = self.x, self.xb, self.xb2, self.hb, self.hb2
        q, qc, qg = self.q, self.qc, self.qg
        rot = self.rope[pos]
        T = pos + 1
        if T > self.used:
            self.used = T

        x[:] = self.tok_emb[token]

        for l, (rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2) in enumerate(self.layers):
            # attention rmsnorm: weight * (ss * x), ss = 1/sqrt(sum(x^2)/dim + 1e-5)
            ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
            multiply(x, ss, out=xb)
            multiply(rms_att, xb, out=xb)

            kc = self.key_cache[l]
            vc = self.value_cache[l]
            kslot = kc[:, pos, :]
            vecdot(wq, xb, out=q)
            vecdot(wk, xb, out=kslot)          # k written straight into the cache slot
            vecdot(wv, xb, out=vc[:, pos, :])  # v likewise

            # RoPE on q and on the cached k
            multiply(qc, rot, out=qc)
            kcx = kslot.view(np.complex64)
            multiply(kcx, rot, out=kcx)

            # multihead attention over positions 0..pos
            att = np.matmul(qg, kc[:, :T, :].transpose(0, 2, 1))  # (kv_heads, kv_mul, T)
            att /= sqrt_hs
            att -= att.max(axis=-1, keepdims=True)
            exp(att, out=att)
            att /= att.sum(axis=-1, keepdims=True, dtype=F32)
            np.matmul(att, vc[:, :T, :], out=xb.reshape(qg.shape))

            vecdot(wo, xb, out=xb2)
            x += xb2

            # ffn rmsnorm
            ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
            multiply(x, ss, out=xb)
            multiply(rms_ffn, xb, out=xb)
            vecdot(w1, xb, out=hb)
            vecdot(w3, xb, out=hb2)
            # SwiGLU: hb = hb * (1 / (1 + exp(-hb))) * hb2
            sig = exp(np.negative(hb))
            sig += one
            np.divide(one, sig, out=sig)
            hb *= sig
            hb *= hb2
            vecdot(w2, hb, out=xb)
            x += xb

        ss = one / np.sqrt(vecdot(x, x) / fdim + eps)
        multiply(x, ss, out=x)
        multiply(self.rms_final, x, out=x)
        vecdot(self.wcls, x, out=self.logits)
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
