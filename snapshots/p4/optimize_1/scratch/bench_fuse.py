import os, sys, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def sep(q, k, v, x, wq, wk, wv):
    for l in range(wq.shape[0]):
        run.matmul_t(q, x, wq[l]); run.matmul_t(k, x, wk[l]); run.matmul_t(v, x, wv[l])
@njit(boundscheck=False)
def fused(qkv, x, w):
    for l in range(w.shape[0]):
        run.matmul_t(qkv, x, w[l])
@njit(boundscheck=False)
def cls_plain(out, x, w):
    run.matmul_t4(out, x, w)
@njit(boundscheck=False)
def cls_tiled(out, x, wt):
    T = wt.shape[2]
    for t in range(wt.shape[0]):
        run.matmul_t(out[t * T:(t + 1) * T], x, wt[t])
m = run.Model('model.bin')
x = np.random.default_rng(1).standard_normal(m.dim).astype(F32)
wqkv = np.concatenate([m.wq, m.wk, m.wv], axis=2)
w13 = np.concatenate([m.w1, m.w3], axis=2)
wt = np.ascontiguousarray(m.wcls.reshape(m.dim, 25, 1280).transpose(1, 0, 2))
q = np.zeros(768, F32); k = np.zeros(768, F32); v = np.zeros(768, F32); qkv = np.zeros(2304, F32)
h1 = np.zeros(2048, F32); h3 = np.zeros(2048, F32); h13 = np.zeros(4096, F32)
lo = np.zeros(32000, F32); lo2 = np.zeros(32000, F32)
cases = {
    'qkv separate': (lambda: sep(q, k, v, x, m.wq, m.wk, m.wv), 3 * m.wq.nbytes),
    'qkv fused':    (lambda: fused(qkv, x, wqkv), wqkv.nbytes),
    'w1,w3 separate': (lambda: (fused(h1, x, m.w1), fused(h3, x, m.w3)), 2 * m.w1.nbytes),
    'w13 fused':    (lambda: fused(h13, x, w13), w13.nbytes),
    'cls plain u4': (lambda: cls_plain(lo, x, m.wcls), m.wcls.nbytes),
    'cls tiled 1280': (lambda: cls_tiled(lo2, x, wt), m.wcls.nbytes),
}
R = {c: [] for c in cases}
for f, _ in cases.values(): f()
for _ in range(15):
    for c, (f, _) in cases.items():
        t = time.perf_counter(); f(); R[c].append(time.perf_counter() - t)
for c, (f, nb) in cases.items():
    print(f'{c:16s} median {statistics.median(R[c])*1e3:7.3f} ms  min {min(R[c])*1e3:7.3f} ms  {nb/min(R[c])/1e9:5.1f} GB/s')
print('qkv identical', np.array_equal(qkv, np.concatenate([q, k, v])), 'w13 identical', np.array_equal(h13, np.concatenate([h1, h3])), 'cls identical', np.array_equal(lo, lo2))
