import os, sys, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def layers_tiled(out, x, wt):  # wt: (L, nt, n, T)
    T = wt.shape[3]
    for l in range(wt.shape[0]):
        for t in range(wt.shape[1]):
            run.matmul_t(out[t * T:(t + 1) * T], x, wt[l, t])
@njit(boundscheck=False)
def layers_plain(out, x, w):
    for l in range(w.shape[0]):
        run.matmul_t(out, x, w[l])
m = run.Model('model.bin')
def tile(w, T):  # (L, n, d) -> (L, d/T, n, T)
    L, n, d = w.shape
    return np.ascontiguousarray(w.reshape(L, n, d // T, T).transpose(0, 2, 1, 3))
cases = {}
for name, w in (('w1 768->2048', m.w1), ('w2 2048->768', m.w2), ('wq 768->768', m.wq)):
    n, d = w.shape[1], w.shape[2]
    x = np.random.default_rng(2).standard_normal(n).astype(F32)
    ref = np.zeros(d, F32); layers_plain(ref, x, w)
    cases[f'{name} plain'] = (layers_plain, (np.zeros(d, F32), x, w), w.nbytes, ref)
    for T in (128, 256, 384, 512, 1024):
        if d % T == 0 and T < d:
            cases[f'{name} T={T}'] = (layers_tiled, (np.zeros(d, F32), x, tile(w, T)), w.nbytes, ref)
R = {c: [] for c in cases}
for f, a, _, _ in cases.values(): f(*a)
for _ in range(12):
    for c, (f, a, _, _) in cases.items():
        t = time.perf_counter(); f(*a); R[c].append(time.perf_counter() - t)
for c, (f, a, nb, ref) in cases.items():
    print(f'{c:22s} median {statistics.median(R[c])*1e3:7.3f} ms  min {min(R[c])*1e3:7.3f}  {nb/min(R[c])/1e9:5.1f} GB/s  same={np.array_equal(a[0], ref)}')
