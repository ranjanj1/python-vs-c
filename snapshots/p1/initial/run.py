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
            data = array('f')
            data.frombytes(f.read())
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
        self.key_cache = array('f', bytes(4 * L * p.seq_len * kv_dim))
        self.value_cache = array('f', bytes(4 * L * p.seq_len * kv_dim))

        # RoPE: freq per even index i, as 1.0f / powf(10000.0f, head_dim / (float)head_size)
        self.rope_freq = []
        for i in range(0, dim, 2):
            head_dim = i % head_size
            e = f32(head_dim / head_size)
            self.rope_freq.append(f32(1.0 / f32(10000.0 ** e)))
        self.inv_sqrt_hs = f32(math.sqrt(head_size))

    def reset(self):
        """Clean KV cache (each command starts from position 0)."""
        n = len(self.key_cache)
        zero = bytes(4 * n)
        self.key_cache = array('f', zero)
        self.value_cache = array('f', zero)

    # -- blocks ---------------------------------------------------------------

    def matmul(self, x, off, n, d):
        """W (d,n) @ x (n,) -> (d,), W stored row-major at data[off:]."""
        w = self.data
        out = [0.0] * d
        for i in range(d):
            base = off + i * n
            val = 0.0
            for a, b in zip(w[base:base + n], x):
                val += a * b
            out[i] = val
        return rf32(out)

    def rmsnorm(self, x, off):
        size = len(x)
        ss = 0.0
        for v in x:
            ss += v * v
        ss = f32(ss)
        ss = f32(ss / size)
        ss = f32(ss + f32(1e-5))
        ss = f32(1.0 / f32(math.sqrt(ss)))
        w = self.data[off:off + size]
        return rf32([w[j] * f32(ss * x[j]) for j in range(size)])

    def forward(self, token, pos):
        dim, hd = self.dim, self.hidden_dim
        kv_dim, kv_mul, hs = self.kv_dim, self.kv_mul, self.head_size
        seq_len, n_heads = self.seq_len, self.n_heads
        data = self.data
        kc, vc = self.key_cache, self.value_cache

        e = self.o_emb + token * dim
        x = data[e:e + dim].tolist()

        # RoPE cos/sin for this position (float32 like cosf/sinf of float32 arg)
        rope = []
        for freq in self.rope_freq:
            val = f32(pos * freq)
            rope.append((f32(math.cos(val)), f32(math.sin(val))))

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
            k = array('f', k)

            # store k, v in cache at this position
            loff = l * seq_len * kv_dim
            co = loff + pos * kv_dim
            kc[co:co + kv_dim] = k
            vc[co:co + kv_dim] = array('f', v)

            # multihead attention
            xb = [0.0] * dim
            scale = self.inv_sqrt_hs
            npos = pos + 1
            for h in range(n_heads):
                qo = h * hs
                qh = q[qo:qo + hs]
                ko = loff + (h // kv_mul) * hs
                att = [0.0] * npos
                for t in range(npos):
                    b = ko + t * kv_dim
                    score = 0.0
                    for a, c in zip(qh, kc[b:b + hs]):
                        score += a * c
                    att[t] = f32(score) / scale
                att = rf32(att)
                # softmax
                mx = att[0]
                for t in range(1, npos):
                    if att[t] > mx:
                        mx = att[t]
                s = 0.0
                for t in range(npos):
                    ev = f32(math.exp(f32(att[t] - mx)))
                    att[t] = ev
                    s += ev
                s = f32(s)
                att = rf32([a / s for a in att])
                # weighted sum of values
                acc = [0.0] * hs
                for t in range(npos):
                    b = ko + t * kv_dim
                    a = att[t]
                    j = 0
                    for c in vc[b:b + hs]:
                        acc[j] += a * c
                        j += 1
                xb[qo:qo + hs] = rf32(acc)

            xb2 = self.matmul(xb, self.o_wo + l * dim * dim, dim, dim)
            x = rf32([x[i] + xb2[i] for i in range(dim)])

            xb = self.rmsnorm(x, self.o_rms_ffn + l * dim)
            hb = self.matmul(xb, self.o_w1 + l * dim * hd, dim, hd)
            hb2 = self.matmul(xb, self.o_w3 + l * dim * hd, dim, hd)
            for i in range(hd):
                val = hb[i]
                if val < -88.0:
                    sig = 0.0  # expf(-val) overflows to inf in float32
                else:
                    sig = f32(1.0 / f32(1.0 + f32(math.exp(-val))))
                hb[i] = f32(f32(val * sig) * hb2[i])
            xb = self.matmul(hb, self.o_w2 + l * dim * hd, hd, dim)
            x = rf32([x[i] + xb[i] for i in range(dim)])

        x = self.rmsnorm(x, self.o_rms_final)
        return array('f', self.matmul(x, self.o_wcls, dim, self.vocab_size))


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
