import numpy as np, time
from numba import njit
F32 = np.float32
import run

@njit(boundscheck=False)
def mm8(out, x, wt):
    n = wt.shape[0]; d = wt.shape[1]
    for i in range(d): out[i] = F32(0.0)
    j = 0
    while j + 8 <= n:
        x0=x[j];x1=x[j+1];x2=x[j+2];x3=x[j+3];x4=x[j+4];x5=x[j+5];x6=x[j+6];x7=x[j+7]
        w0=wt[j];w1=wt[j+1];w2=wt[j+2];w3=wt[j+3];w4=wt[j+4];w5=wt[j+5];w6=wt[j+6];w7=wt[j+7]
        for i in range(d):
            v = out[i]
            v += w0[i]*x0; v += w1[i]*x1; v += w2[i]*x2; v += w3[i]*x3
            v += w4[i]*x4; v += w5[i]*x5; v += w6[i]*x6; v += w7[i]*x7
            out[i] = v
        j += 8
    while j < n:
        xj = x[j]; wj = wt[j]
        for i in range(d): out[i] += wj[i]*xj
        j += 1

@njit(boundscheck=False)
def mm2(out, x, wt):
    n = wt.shape[0]; d = wt.shape[1]
    for i in range(d): out[i] = F32(0.0)
    j = 0
    while j + 2 <= n:
        x0=x[j];x1=x[j+1]; w0=wt[j];w1=wt[j+1]
        for i in range(d):
            v = out[i]; v += w0[i]*x0; v += w1[i]*x1; out[i] = v
        j += 2
    while j < n:
        xj = x[j]; wj = wt[j]
        for i in range(d): out[i] += wj[i]*xj
        j += 1

# 12 layers worth of distinct weights so caches can't hold them (real access pattern)
rng = np.random.default_rng(0)
for n, d in [(768, 768), (768, 2048), (2048, 768), (768, 32000)]:
    L = 12 if d != 32000 else 2
    W = rng.standard_normal((L, n, d), dtype=np.float32)
    x = rng.standard_normal(n).astype(F32); out = np.zeros(d, F32)
    res = {}
    for name, f in (('u2', mm2), ('u4', run.matmul_t), ('u8', mm8)):
        f(out, x, W[0]); best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for l in range(L): f(out, x, W[l])
            best = min(best, (time.perf_counter() - t) / L)
        res[name] = out.copy()
        print(f'{n}x{d} {name} {best*1e3:.3f} ms {n*d*4/best/1e9:.1f} GB/s')
    print('  identical:', all(np.array_equal(res['u4'], r) for r in res.values()))
