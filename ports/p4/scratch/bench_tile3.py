import os, sys, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def tile8(out, x, wt):
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
def L4(out, x, wt):
    for l in range(wt.shape[0]): run.matvec_tiles(out, x, wt[l])
@njit(boundscheck=False)
def L8(out, x, wt):
    T = wt.shape[3]
    for l in range(wt.shape[0]):
        for t in range(wt.shape[1]): tile8(out[t*T:(t+1)*T], x, wt[l, t])
m = run.Model('model.bin')
w13 = m.w13  # (L, 32, 768, 128)
L, nt, n, T = w13.shape
dense = w13.transpose(0, 2, 1, 3).reshape(L, n, nt * T)  # (L, n, 4096) view
def retile(Tn):
    return np.ascontiguousarray(dense.reshape(L, n, nt * T // Tn, Tn).transpose(0, 2, 1, 3))
del m
x = np.random.default_rng(3).standard_normal(n).astype(F32)
cases = {}
for Tn in (64, 128, 256):
    wt = retile(Tn)
    cases[f'T={Tn} u4'] = (L4, (np.zeros(nt*T, F32), x, wt))
    cases[f'T={Tn} u8'] = (L8, (np.zeros(nt*T, F32), x, wt))
R = {c: [] for c in cases}
for f, a in cases.values(): f(*a)
for _ in range(12):
    for c, (f, a) in cases.items():
        t = time.perf_counter(); f(*a); R[c].append(time.perf_counter() - t)
ref = cases['T=128 u4'][1][0]
for c, (f, a) in cases.items():
    print(f'w1|w3 {c:10s} median {statistics.median(R[c])*1e3:7.3f} ms  min {min(R[c])*1e3:7.3f}  {w13.nbytes/min(R[c])/1e9:5.1f} GB/s same={np.array_equal(a[0], ref)}')
