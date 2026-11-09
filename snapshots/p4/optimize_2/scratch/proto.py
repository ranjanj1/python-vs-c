"""Prototype forward variants vs run.forward on the dev sequence: bitwise equality + interleaved timing."""
import math, os, sys, time, statistics, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
matmul_t = run.matmul_t
softmax = run.softmax
rmsnorm = run.rmsnorm


@njit(boundscheck=False)
def attn_v1(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs):
    kv_mul = n_heads // n_kv_heads
    n_t = pos + 1
    for h in range(n_heads):
        hk = h // kv_mul; kh = kc[hk]; a = att[h]
        for t in range(n_t): a[t] = F32(0.0)
        qo = h * head_size
        for i in range(head_size):
            qi = q[qo + i]; kr = kh[i]
            for t in range(n_t): a[t] += qi * kr[t]
        for t in range(n_t): a[t] = a[t] / sqrt_hs
        softmax(a, n_t)
        xo = h * head_size; vo = hk * head_size
        for i in range(head_size): xb[xo + i] = F32(0.0)
        for t in range(n_t):
            at = a[t]; vr = vc[t]
            for i in range(head_size): xb[xo + i] += at * vr[vo + i]


@njit(boundscheck=False)
def attn_v2(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs):
    # scores: two key rows per pass (same i order per score); values: two timesteps per pass
    kv_mul = n_heads // n_kv_heads
    n_t = pos + 1
    for h in range(n_heads):
        hk = h // kv_mul; kh = kc[hk]; a = att[h]
        qo = h * head_size
        q0 = q[qo]; k0 = kh[0]
        for t in range(n_t): a[t] = F32(0.0) + q0 * k0[t]
        i = 1
        while i + 2 <= head_size:
            qa = q[qo + i]; qb = q[qo + i + 1]; ka = kh[i]; kb = kh[i + 1]
            for t in range(n_t):
                s = a[t]; s += qa * ka[t]; s += qb * kb[t]; a[t] = s
            i += 2
        while i < head_size:
            qi = q[qo + i]; kr = kh[i]
            for t in range(n_t): a[t] += qi * kr[t]
            i += 1
        for t in range(n_t): a[t] = a[t] / sqrt_hs
        softmax(a, n_t)
        xo = h * head_size; vo = hk * head_size
        o = xb[xo:xo + head_size]
        for i in range(head_size): o[i] = F32(0.0)
        t = 0
        while t + 2 <= n_t:
            aa = a[t]; ab = a[t + 1]; va = vc[t, vo:vo + head_size]; vb = vc[t + 1, vo:vo + head_size]
            for i in range(head_size):
                s = o[i]; s += aa * va[i]; s += ab * vb[i]; o[i] = s
            t += 2
        while t < n_t:
            aa = a[t]; va = vc[t, vo:vo + head_size]
            for i in range(head_size): o[i] += aa * va[i]
            t += 1


@njit(boundscheck=False)
def fwd2(token, pos, emb, rms_att, wqkv, wo, rms_ffn, w13, w2, rms_final, wcls_tiles,
         key_cache, value_cache, x, xb, xb2, qkv, h13, att, logits, rope_c, rope_s, n_heads, n_kv_heads, attv):
    dim = x.shape[0]
    hidden_dim = h13.shape[0] // 2
    n_layers = wqkv.shape[0]
    head_size = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads
    hs_f = F32(head_size)
    sqrt_hs = math.sqrt(hs_f)
    for i in range(dim):
        x[i] = emb[token, i]
    # RoPE coefficients depend only on (i, pos): same expressions as run.c, computed once per position
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = F32(1.0) / math.pow(F32(10000.0), F32(head_dim) / hs_f)
        val = F32(pos) * freq
        rope_c[i // 2] = math.cos(val)
        rope_s[i // 2] = math.sin(val)
    q = qkv[:dim]; k = qkv[dim:dim + kv_dim]; v = qkv[dim + kv_dim:]
    for l in range(n_layers):
        rmsnorm(xb, x, rms_att[l])
        matmul_t(qkv, xb, wqkv[l])
        for i in range(0, dim, 2):
            fcr = rope_c[i // 2]; fci = rope_s[i // 2]
            v0 = q[i]; v1 = q[i + 1]
            q[i] = v0 * fcr - v1 * fci; q[i + 1] = v0 * fci + v1 * fcr
            if i < kv_dim:
                v0 = k[i]; v1 = k[i + 1]
                k[i] = v0 * fcr - v1 * fci; k[i + 1] = v0 * fci + v1 * fcr
        kc = key_cache[l]; vc = value_cache[l]
        for hk in range(n_kv_heads):
            for i in range(head_size):
                kc[hk, i, pos] = k[hk * head_size + i]
        for i in range(kv_dim):
            vc[pos, i] = v[i]
        if attv == 1:
            attn_v1(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs)
        else:
            attn_v2(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs)
        matmul_t(xb2, xb, wo[l])
        for i in range(dim):
            x[i] += xb2[i]
        rmsnorm(xb, x, rms_ffn[l])
        matmul_t(h13, xb, w13[l])
        for i in range(hidden_dim):
            val = h13[i]
            val *= F32(1.0) / (F32(1.0) + math.exp(-val))
            val *= h13[hidden_dim + i]
            h13[i] = val
        matmul_t(xb, h13[:hidden_dim], w2[l])
        for i in range(dim):
            x[i] += xb[i]
    rmsnorm(x, x, rms_final)
    T = wcls_tiles.shape[2]
    for t in range(wcls_tiles.shape[0]):
        matmul_t(logits[t * T:(t + 1) * T], x, wcls_tiles[t])


m = run.Model('model.bin')
L, dim, hid, V = m.n_layers, m.dim, m.hidden_dim, m.vocab
wqkv = np.concatenate([m.wq, m.wk, m.wv], axis=2)
w13 = np.concatenate([m.w1, m.w3], axis=2)
T = 1280
wcls_tiles = np.ascontiguousarray(m.wcls.reshape(dim, V // T, T).transpose(1, 0, 2))
qkv = np.zeros(wqkv.shape[2], F32); h13 = np.zeros(2 * hid, F32)
rope_c = np.zeros(dim // 2, F32); rope_s = np.zeros(dim // 2, F32)
logits2 = np.zeros(V, F32)
fx = json.load(open(os.path.join(os.path.dirname(run.__file__), 'fixtures.json')))
seq = fx['prompts'][2]['seq']


def run_old():
    m.reset(); out = np.empty((len(seq), V), F32)
    for p, t in enumerate(seq):
        out[p] = m.forward(t, p)
    return out


def run_new(attv):
    m.reset(); out = np.empty((len(seq), V), F32)
    for p, t in enumerate(seq):
        fwd2(t, p, m.emb, m.rms_att, wqkv, m.wo, m.rms_ffn, w13, m.w2, m.rms_final, wcls_tiles,
             m.key_cache, m.value_cache, m.x, m.xb, m.xb2, qkv, h13, m.att, logits2, rope_c, rope_s,
             m.n_heads, m.n_kv_heads, attv)
        out[p] = logits2
    return out


a = run_old(); b1 = run_new(1); b2 = run_new(2)
print('new(v1 attention) bitwise identical:', np.array_equal(a, b1))
print('new(v2 attention) max |diff| vs old:', float(np.abs(a - b2).max()), 'identical:', np.array_equal(a, b2))
R = {'old': [], 'new attn v1': [], 'new attn v2': []}
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
    for name, f in (('old', run_old), ('new attn v1', lambda: run_new(1)), ('new attn v2', lambda: run_new(2))):
        t = time.perf_counter(); f(); R[name].append((time.perf_counter() - t) / len(seq))
for k, v in R.items():
    print(f'{k:12s} median {statistics.median(v)*1e3:.3f} ms/token  min {min(v)*1e3:.3f}')
