"""Hand-written numerical control, not an entrant. Same contract as the ports. DTYPE float32 or float64 (argv[3]).

Purpose: a second, independently written implementation to check the C reference and to size the numerical tolerance.
Tokenizer work is not repeated here: ENCODE, PIECE and GENERATE are not implemented.
"""
import struct, sys
import numpy as np

dt = np.dtype(sys.argv[3] if len(sys.argv) > 3 else 'float32')
raw = open(sys.argv[1], 'rb').read()
dim, hid, L, H, KV, V0, S = struct.unpack('7i', raw[:28])
shared, V = V0 > 0, abs(V0)
w = np.frombuffer(raw, np.float32, offset=28)
hs = dim // H; kv = dim * KV // H; o = 0
def take(*shape):
    global o
    n = int(np.prod(shape)); a = w[o:o + n].reshape(shape).astype(dt); o += n; return a
emb = take(V, dim); att_n = take(L, dim); wq = take(L, dim, dim); wk = take(L, kv, dim); wv = take(L, kv, dim); wo = take(L, dim, dim)
ffn_n = take(L, dim); w1 = take(L, hid, dim); w2 = take(L, dim, hid); w3 = take(L, hid, dim); fin = take(dim)
o += S * hs  # legacy rope tables
wcls = emb if shared else take(V, dim)
kc = np.zeros((L, S, kv), dt); vc = np.zeros((L, S, kv), dt)

def rms(x, g):
    return g * (x / np.sqrt(np.mean(x * x, dtype=dt) + dt.type(1e-5)))

def forward(tok, pos):
    x = emb[tok].copy()
    i = np.arange(0, hs, 2, dtype=dt)
    ang = pos * (dt.type(1) / dt.type(10000) ** (i / dt.type(hs))).astype(dt)
    c, s = np.cos(ang).astype(dt), np.sin(ang).astype(dt)
    def rot(v):
        v = v.reshape(-1, hs // 2, 2); a, b = v[..., 0], v[..., 1]
        return np.stack([a * c - b * s, a * s + b * c], -1).reshape(-1)
    for l in range(L):
        h = rms(x, att_n[l]); q = rot(wq[l] @ h); k = rot(wk[l] @ h); v = wv[l] @ h
        kc[l, pos] = k; vc[l, pos] = v
        out = np.empty(dim, dt); rep = H // KV
        for a in range(H):
            b = (a // rep) * hs
            sc = kc[l, :pos + 1, b:b + hs] @ q[a * hs:(a + 1) * hs] / np.sqrt(dt.type(hs))
            e = np.exp(sc - sc.max()); e /= e.sum()
            out[a * hs:(a + 1) * hs] = e @ vc[l, :pos + 1, b:b + hs]
        x = x + wo[l] @ out
        h = rms(x, ffn_n[l]); g = w1[l] @ h
        x = x + w2[l] @ ((g / (1 + np.exp(-g))) * (w3[l] @ h))
    return wcls @ rms(x, fin)

print('READY', flush=True)
for line in sys.stdin:
    p = line.split()
    if p[0] == 'QUIT':
        break
    if p[0] == 'LOGITS':
        ids = list(map(int, p[2:]))
        np.stack([forward(t, i) for i, t in enumerate(ids)]).astype('<f4').tofile(p[1])
        print('DONE', len(ids), flush=True)
    elif p[0] == 'FIXED':
        ids = list(map(int, p[1:]))
        print('DONE', len(ids), *[int(forward(t, i).argmax()) for i, t in enumerate(ids)], flush=True)
    else:
        print('ERROR unsupported', flush=True)
