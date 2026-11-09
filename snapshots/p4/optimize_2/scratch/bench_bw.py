import numpy as np, time, statistics
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def stream(out, w):  # out[i] += w[j, i]: pure streaming read, vectorized, out in L1
    for j in range(w.shape[0]):
        r = w[j]
        for i in range(w.shape[1]):
            out[i] += r[i]
@njit(boundscheck=False)
def stream2(out, w):
    j = 0
    while j + 2 <= w.shape[0]:
        r0 = w[j]; r1 = w[j + 1]
        for i in range(w.shape[1]):
            v = out[i]; v += r0[i]; v += r1[i]; out[i] = v
        j += 2
a = np.ones((100000, 1024), F32)  # 410 MB, like one token's weights
out = np.zeros(1024, F32); stream(out, a[:10]); stream2(out, a[:10])
for name, f in (('numba stream u1', lambda: stream(out, a)), ('numba stream u2', lambda: stream2(out, a)), ('numpy sum', lambda: a.sum())):
    ts = []
    for _ in range(7):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    print(f'{name:16s} min {a.nbytes/min(ts)/1e9:.1f} GB/s  median {a.nbytes/statistics.median(ts)/1e9:.1f} GB/s')
