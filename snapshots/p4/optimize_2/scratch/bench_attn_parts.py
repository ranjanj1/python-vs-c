"""Attention at long context: time of the softmax (expf) part alone, 144 rows (12 layers x 12 heads)."""
import os, sys, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def softmax_rows(att, n, reps):
    for _ in range(reps):
        for h in range(att.shape[0]):
            run.softmax(att[h], n)
att = (np.random.default_rng(0).standard_normal((144, 1024)) * 3).astype(F32)
softmax_rows(att, 10, 1)
for n in (80, 300, 1000):
    ts = []
    for _ in range(9):
        a = att.copy(); t = time.perf_counter(); softmax_rows(a, n, 1); ts.append(time.perf_counter() - t)
    print(f'softmax over {n:4d} positions x 144 rows: {statistics.median(ts)*1e3:.3f} ms per token')
