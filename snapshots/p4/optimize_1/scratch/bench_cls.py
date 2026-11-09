import sys, os, time, statistics, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32

@njit(boundscheck=False)
def cls_tiled(out, x, wtiles, k):
    # wtiles: (n_tiles, n, T) ; tile t covers outputs t*T .. t*T+T-1
    nt = wtiles.shape[0]; T = wtiles.shape[2]
    for t in range(nt):
        o = out[t * T:(t + 1) * T]
        if k == 2: run.matmul_t(o, x, wtiles[t])
        else: run.matmul_t4(o, x, wtiles[t])

m = run.Model('model.bin')
x = np.random.default_rng(0).standard_normal(m.dim).astype(F32)
ref = np.zeros(m.vocab, F32); run.matmul_t4(ref, x, m.wcls)
def bench(f, *a):
    f(*a); ts = []
    for _ in range(15):
        t = time.perf_counter(); f(*a); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)
md, mn = bench(run.matmul_t4, ref, x, m.wcls)
print(f'untiled u4: median {md*1e3:.3f} ms  min {mn*1e3:.3f}  {m.wcls.nbytes/mn/1e9:.1f} GB/s')
for T in (500, 1000, 2000, 4000, 8000):
    wt = np.ascontiguousarray(m.wcls.reshape(m.dim, m.vocab // T, T).transpose(1, 0, 2))
    out = np.zeros(m.vocab, F32)
    for k in (2, 4):
        md, mn = bench(cls_tiled, out, x, wt, k)
        print(f'T={T:5d} u{k}: median {md*1e3:.3f} ms  min {mn*1e3:.3f}  {m.wcls.nbytes/mn/1e9:.1f} GB/s  identical={np.array_equal(out, ref)}')
