import os, sys, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def mm4(out, x, wt):
    n = wt.shape[0]; d = wt.shape[1]
    for i in range(d): out[i] = F32(0.0)
    j = 0
    while j + 4 <= n:
        x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3]
        w0 = wt[j]; w1 = wt[j+1]; w2 = wt[j+2]; w3 = wt[j+3]
        for i in range(d):
            v = out[i]; v += w0[i]*x0; v += w1[i]*x1; v += w2[i]*x2; v += w3[i]*x3; out[i] = v
        j += 4
    while j < n:
        xj = x[j]; wj = wt[j]
        for i in range(d): out[i] += wj[i]*xj
        j += 1
@njit(boundscheck=False)
def mm1(out, x, wt):
    n = wt.shape[0]; d = wt.shape[1]
    for i in range(d): out[i] = F32(0.0)
    for j in range(n):
        xj = x[j]; wj = wt[j]
        for i in range(d): out[i] += wj[i]*xj
@njit(boundscheck=False)
def tiled(out, x, wt, k):  # wt (L, nt, n, T)
    T = wt.shape[3]
    for l in range(wt.shape[0]):
        for t in range(wt.shape[1]):
            o = out[t*T:(t+1)*T]
            if k == 1: mm1(o, x, wt[l, t])
            elif k == 2: run.matmul_t(o, x, wt[l, t])
            else: mm4(o, x, wt[l, t])
m = run.Model('model.bin')
def tile(w, T):
    L, n, d = w.shape
    return np.ascontiguousarray(w.reshape(L, n, d // T, T).transpose(0, 2, 1, 3))
cases = {}
for name, w in (('w1', m.w1), ('w2', m.w2), ('cls', m.wcls[None])):
    n, d = w.shape[1], w.shape[2]
    x = np.random.default_rng(2).standard_normal(n).astype(F32)
    for T in (128, 256):
        wt = tile(w, T)
        for k in (1, 2, 4):
            cases[f'{name} T={T} u{k}'] = (tiled, (np.zeros(d, F32), x, wt, k), w.nbytes)
    del wt
R = {c: [] for c in cases}
for f, a, _ in cases.values(): f(*a)
for _ in range(10):
    for c, (f, a, _) in cases.items():
        t = time.perf_counter(); f(*a); R[c].append(time.perf_counter() - t)
for c, (f, a, nb) in cases.items():
    print(f'{c:16s} median {statistics.median(R[c])*1e3:7.3f} ms  min {min(R[c])*1e3:7.3f}  {nb/min(R[c])/1e9:5.1f} GB/s')
