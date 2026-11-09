# vecdot throughput vs row length (cold-ish: 48 MB working set > caches), fixed total bytes.
import time, numpy as np
F = np.float32; pc = time.perf_counter
tot = 12 * 1024 * 1024  # floats (48 MB)
for n in (64, 128, 256, 300, 384, 512, 600, 700, 768, 800, 1000, 1024, 1536, 2048, 4096):
    rows = tot // n; W = np.ones((rows, n), F); x = np.ones(n, F); o = np.empty(rows, F)
    best = 1e9
    for r in range(5):
        t = pc(); np.vecdot(W, x, out=o); best = min(best, pc() - t)
    best2 = 1e9
    for r in range(5):
        t = pc(); np.dot(W, x, out=o); best2 = min(best2, pc() - t)
    print('n=%5d  vecdot %5.1f GB/s   gemv %5.1f GB/s' % (n, tot * 4 / best / 1e9, tot * 4 / best2 / 1e9))
