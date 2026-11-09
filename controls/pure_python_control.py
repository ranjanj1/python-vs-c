"""Hand-written plain-Python control, not an entrant. Standard library only, runs on CPython and PyPy. FIXED and LOGITS only.

Used to set the warm-up and time-limit policy from real pilot series. Python floats are 64-bit, so results are not float32 arithmetic.
"""
import array, math, struct, sys
from operator import mul

raw = open(sys.argv[1], 'rb').read()
dim, hid, L, H, KV, V0, S = struct.unpack('7i', raw[:28])
V = abs(V0); hs = dim // H; kvd = dim * KV // H
w = array.array('f'); w.frombytes(raw[28:])
del raw
o = 0
def take(n):
    global o
    a = w[o:o + n]; o += n; return a
def mat(rows, cols):
    return [take(cols) for _ in range(rows)]
emb = mat(V, dim); att = [take(dim) for _ in range(L)]
wq = [mat(dim, dim) for _ in range(L)]; wk = [mat(kvd, dim) for _ in range(L)]; wv = [mat(kvd, dim) for _ in range(L)]; wo = [mat(dim, dim) for _ in range(L)]
ffn = [take(dim) for _ in range(L)]; w1 = [mat(hid, dim) for _ in range(L)]; w2 = [mat(dim, hid) for _ in range(L)]; w3 = [mat(hid, dim) for _ in range(L)]
fin = take(dim)
del w

def mv(m, x):
    return [sum(map(mul, r, x)) for r in m]

def rms(x, g):
    s = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
    return [a * b * s for a, b in zip(g, x)]

kc = [[None] * S for _ in range(L)]; vc = [[None] * S for _ in range(L)]

def rope(v, pos):
    for i in range(0, len(v), 2):
        f = 1.0 / 10000.0 ** ((i % hs) / hs); c = math.cos(pos * f); s = math.sin(pos * f)
        a, b = v[i], v[i + 1]; v[i] = a * c - b * s; v[i + 1] = a * s + b * c

def forward(tok, pos):
    x = list(emb[tok])
    for l in range(L):
        h = rms(x, att[l]); q = mv(wq[l], h); k = mv(wk[l], h); v = mv(wv[l], h)
        rope(q, pos); rope(k, pos); kc[l][pos] = k; vc[l][pos] = v
        out = [0.0] * dim
        for a in range(H):
            b = a * hs; qa = q[b:b + hs]
            sc = [sum(map(mul, qa, kc[l][t][b:b + hs])) / math.sqrt(hs) for t in range(pos + 1)]
            m = max(sc); e = [math.exp(z - m) for z in sc]; d = sum(e)
            for t in range(pos + 1):
                c = e[t] / d; vt = vc[l][t]
                for i in range(hs):
                    out[b + i] += c * vt[b + i]
        x = [p + r for p, r in zip(x, mv(wo[l], out))]
        h = rms(x, ffn[l]); g = mv(w1[l], h); u = mv(w3[l], h)
        y = mv(w2[l], [p / (1.0 + math.exp(-p)) * r for p, r in zip(g, u)])
        x = [p + r for p, r in zip(x, y)]
    return mv(emb, rms(x, fin))

print('READY', flush=True)
for line in sys.stdin:
    p = line.split()
    if p[0] == 'QUIT':
        break
    if p[0] == 'FIXED':
        ids = list(map(int, p[1:]))
        am = []
        for i, t in enumerate(ids):
            lg = forward(t, i); am.append(lg.index(max(lg)))
        print('DONE', len(ids), *am, flush=True)
    elif p[0] == 'LOGITS':
        ids = list(map(int, p[2:]))
        with open(p[1], 'wb') as f:
            for i, t in enumerate(ids):
                array.array('f', forward(t, i)).tofile(f)
        print('DONE', len(ids), flush=True)
    else:
        print('ERROR unsupported', flush=True)
