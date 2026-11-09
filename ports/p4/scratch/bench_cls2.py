import sys, os, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def cls_tiled(out, x, wtiles):
    nt = wtiles.shape[0]; T = wtiles.shape[2]
    for t in range(nt):
        run.matmul_t(out[t * T:(t + 1) * T], x, wtiles[t])
m = run.Model('model.bin')
x = np.random.default_rng(0).standard_normal(m.dim).astype(F32)
Ts = (256, 500, 640, 800, 1280, 1600, 2000, 3200)
W = {T: np.ascontiguousarray(m.wcls.reshape(m.dim, m.vocab // T, T).transpose(1, 0, 2)) for T in Ts}
del m
out = np.zeros(32000, F32); res = {T: [] for T in Ts}
for T in Ts: cls_tiled(out, x, W[T])
for r in range(12):
    for T in Ts:
        t = time.perf_counter(); cls_tiled(out, x, W[T]); res[T].append(time.perf_counter() - t)
for T in Ts:
    print(f'T={T:5d} median {statistics.median(res[T])*1e3:.3f} ms  min {min(res[T])*1e3:.3f} ms  {98.3e6/min(res[T])/1e9:.1f} GB/s')
