"""Key-cache layout: current (KV, hs, S) vs blocked (KV, S/B, hs, B). Store + attention per layer, caches cold."""
import os, sys, time, statistics, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from numba import njit
import run
F32 = np.float32
H = KV = 12; hs = 64; S = 1024; L = 12; B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
sqrt_hs = math.sqrt(F32(hs))

@njit(boundscheck=False)
def store_cur(k, v, kc, vc, pos):
    for hk in range(kc.shape[0]):
        for i in range(hs): kc[hk, i, pos] = k[hk * hs + i]
    for i in range(v.shape[0]): vc[pos, i] = v[i]

@njit(boundscheck=False)
def store_blk(k, v, kc, vc, pos):
    tb = pos // B; c = pos % B
    for hk in range(kc.shape[0]):
        blk = kc[hk, tb]
        for i in range(hs): blk[i, c] = k[hk * hs + i]
    for i in range(v.shape[0]): vc[pos, i] = v[i]

@njit(boundscheck=False)
def attn_blk(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs):
    kv_mul = n_heads // n_kv_heads
    n_t = pos + 1
    Bk = kc.shape[3]
    for h in range(n_heads):
        hk = h // kv_mul; a = att[h]; qo = h * head_size
        for tb in range((n_t + Bk - 1) // Bk):
            kb = kc[hk, tb]; t0 = tb * Bk; n = min(Bk, n_t - t0)
            ab = a[t0:t0 + n]
            q0 = q[qo]; k0 = kb[0]
            for t in range(n): ab[t] = F32(0.0) + q0 * k0[t]
            i = 1
            while i + 2 <= head_size:
                qa = q[qo + i]; qb = q[qo + i + 1]; ka = kb[i]; kb2 = kb[i + 1]
                for t in range(n):
                    s = ab[t]; s += qa * ka[t]; s += qb * kb2[t]; ab[t] = s
                i += 2
            while i < head_size:
                qi = q[qo + i]; kr = kb[i]
                for t in range(n): ab[t] += qi * kr[t]
                i += 1
        for t in range(n_t): a[t] = a[t] / sqrt_hs
        run.softmax(a, n_t)
        vo = hk * head_size; o = xb[qo:qo + head_size]
        for i in range(head_size): o[i] = F32(0.0)
        t = 0
        while t + 2 <= n_t:
            aa = a[t]; ab_ = a[t + 1]; va = vc[t, vo:vo + head_size]; vb = vc[t + 1, vo:vo + head_size]
            for i in range(head_size):
                s = o[i]; s += aa * va[i]; s += ab_ * vb[i]; o[i] = s
            t += 2
        while t < n_t:
            aa = a[t]; va = vc[t, vo:vo + head_size]
            for i in range(head_size): o[i] += aa * va[i]
            t += 1

@njit(boundscheck=False)
def flush(buf):
    s = F32(0)
    for i in range(0, buf.shape[0], 16): s += buf[i]
    return s

rng = np.random.default_rng(0)
kc_c = np.zeros((L, KV, hs, S), F32); kc_b = np.zeros((L, KV, S // B, hs, B), F32); vc = np.zeros((L, S, KV * hs), F32)
vc2 = vc.copy()
att1 = np.zeros((H, S), F32); att2 = np.zeros((H, S), F32); xb1 = np.zeros(H * hs, F32); xb2 = np.zeros(H * hs, F32)
big = np.ones(40_000_000, F32)  # 160 MB to evict L2 between tokens, like the weight stream does
Ks = rng.standard_normal((S, L, KV * hs)).astype(F32); Vs = rng.standard_normal((S, L, KV * hs)).astype(F32)
Q = rng.standard_normal((S, L, H * hs)).astype(F32)
res = {'current': {}, 'blocked': {}}
ok = True
checkpoints = (40, 80, 300, 1000)
def step(name, pos):
    for l in range(L):
        if name == 'current':
            store_cur(Ks[pos, l], Vs[pos, l], kc_c[l], vc[l], pos)
            run.attention(Q[pos, l], kc_c[l], vc[l], att1, xb1, pos, H, KV, hs, sqrt_hs)
        else:
            store_blk(Ks[pos, l], Vs[pos, l], kc_b[l], vc2[l], pos)
            attn_blk(Q[pos, l], kc_b[l], vc2[l], att2, xb2, pos, H, KV, hs, sqrt_hs)
for pos in range(max(checkpoints) + 1):
    step('current', pos); step('blocked', pos)
    ok &= np.array_equal(xb1, xb2)
    if pos in checkpoints:
        for name in ('current', 'blocked') * 9:
            flush(big); t = time.perf_counter(); step(name, pos); res[name].setdefault(pos, []).append(time.perf_counter() - t)
print('blocked == current bitwise at every position:', ok)
for pos in checkpoints:
    c = statistics.median(res['current'][pos]); b = statistics.median(res['blocked'][pos])
    print(f'pos {pos:4d}: store+attention, 12 layers, cold caches (median of 9): current {c*1e3:.3f} ms  blocked {b*1e3:.3f} ms  ({c/b:.2f}x)')
