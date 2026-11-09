#!/usr/bin/env python3
"""P3 port of llama2.c (reference/run.c): NumPy float32 arrays, native BLAS via np.dot / np.matmul.

Usage: python3 run.py <checkpoint.bin> <tokenizer.bin>, then the line protocol from INTERFACE.md on stdin.
"""
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
        # freq = 1 / powf(10000, head_dim / head_size); val = pos * freq; cosf / sinf
        head_dim = np.arange(0, hs, 2, dtype=np.int32).astype(F32)
        freq = (F32(1.0) / np.power(F32(10000.0), head_dim / F32(hs))).astype(F32)
        pos = np.arange(seq_len, dtype=np.int32).astype(F32)
        val = (pos[:, None] * freq[None, :]).astype(F32)
        self.rope_cos = np.cos(val).astype(F32)  # (seq_len, hs/2)
        self.rope_sin = np.sin(val).astype(F32)

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
        self.k = np.zeros(kv_dim, F32)
        self.v = np.zeros(kv_dim, F32)
        self.logits = np.zeros(vocab_size, F32)
        self.inv_sqrt_hs = None
        self.sqrt_hs = F32(math.sqrt(hs))  # sqrtf(head_size) for an exactly representable int

    def reset(self):
        """Clean KV cache: zero every position written by the previous command."""
        if self.used:
            self.key_cache[:, :, :self.used, :] = 0
            self.value_cache[:, :, :self.used, :] = 0
            self.used = 0

    def rmsnorm(self, out, x, weight):
        ss = np.dot(x, x)
        ss = F32(ss) / F32(x.shape[0])
        ss = ss + F32(1e-5)
        ss = F32(1.0) / np.sqrt(ss)
        np.multiply(x, ss, out=out)
        np.multiply(weight, out, out=out)

    def rope(self, vec, pos, n):
        # rotate pairs (vec[i], vec[i+1]) within each head
        c = self.rope_cos[pos]
        s = self.rope_sin[pos]
        v = vec.reshape(n, self.head_size // 2, 2)
        v0 = v[:, :, 0].copy()
        v1 = v[:, :, 1].copy()
        v[:, :, 0] = v0 * c - v1 * s
        v[:, :, 1] = v0 * s + v1 * c

    def forward(self, token, pos):
        dim, hs = self.dim, self.head_size
        H, KVH, kv_mul = self.n_heads, self.n_kv_heads, self.kv_mul
        x, xb, xb2, hb, hb2 = self.x, self.xb, self.xb2, self.hb, self.hb2
        q, k, v = self.q, self.k, self.v
        T = pos + 1
        if T > self.used:
            self.used = T

        x[:] = self.tok_emb[token]

        for l in range(self.n_layers):
            self.rmsnorm(xb, x, self.rms_att[l])

            np.dot(self.wq[l], xb, out=q)
            np.dot(self.wk[l], xb, out=k)
            np.dot(self.wv[l], xb, out=v)

            self.rope(q, pos, H)
            self.rope(k, pos, KVH)

            kc = self.key_cache[l]
            vc = self.value_cache[l]
            kc[:, pos, :] = k.reshape(KVH, hs)
            vc[:, pos, :] = v.reshape(KVH, hs)

            # multihead attention over positions 0..pos
            qh = q.reshape(KVH, kv_mul, hs)              # query heads grouped by their kv head
            keys = kc[:, :T, :]                          # (KVH, T, hs)
            att = np.matmul(qh, keys.transpose(0, 2, 1))  # (KVH, kv_mul, T)
            att /= self.sqrt_hs
            att -= att.max(axis=-1, keepdims=True)
            np.exp(att, out=att)
            att /= att.sum(axis=-1, keepdims=True, dtype=F32)
            out = np.matmul(att, vc[:, :T, :])           # (KVH, kv_mul, hs)
            xb[:] = out.reshape(dim)

            np.dot(self.wo[l], xb, out=xb2)
            x += xb2

            self.rmsnorm(xb, x, self.rms_ffn[l])
            np.dot(self.w1[l], xb, out=hb)
            np.dot(self.w3[l], xb, out=hb2)
            # SwiGLU: hb = hb * (1 / (1 + exp(-hb))) * hb2
            sig = np.negative(hb)
            np.exp(sig, out=sig)
            sig += F32(1.0)
            np.divide(F32(1.0), sig, out=sig)
            hb *= sig
            hb *= hb2
            np.dot(self.w2[l], hb, out=xb)
            x += xb

        self.rmsnorm(x, x, self.rms_final)
        np.dot(self.wcls, x, out=self.logits)
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

        vocab, scores = self.vocab, self.scores
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for j in range(len(tokens) - 1):
                tid = lookup.get(vocab[tokens[j]] + vocab[tokens[j + 1]], -1)
                if tid != -1 and scores[tid] > best_score:
                    best_score = scores[tid]
                    best_id = tid
                    best_idx = j
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]
        return tokens


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
