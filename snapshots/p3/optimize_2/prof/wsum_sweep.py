# Attention weighted sum, 12 layers, cold cache: (t,d) layout via matmul / einsum vs (d,t) layout via vecdot.
import time, numpy as np
F = np.float32; pc = time.perf_counter; rng = np.random.default_rng(0)
L, H, hs, S = 12, 12, 64, 1024
junk = np.ones(64 * 1024 * 1024 // 4, F)
V = rng.standard_normal((L, H, S, hs), dtype=F); Vt = np.ascontiguousarray(V.transpose(0, 1, 3, 2))
def timeit(fn):
    best = 1e9
    for r in range(7):
        tot = 0
        for l in range(L):
            junk.max(); t = pc(); fn(l); tot += pc() - t
        best = min(best, tot)
    return best * 1e3
print('   T  matmul(t,d)  einsum(t,d)  vecdot(d,t)')
for T in range(100, 1025, 100):
    att = rng.random((H, 1, T), dtype=F); out = np.empty((H, 1, hs), F)
    a = timeit(lambda l: np.matmul(att, V[l][:, :T, :], out=out))
    b = timeit(lambda l: np.einsum('hqt,htd->hqd', att, V[l][:, :T, :], out=out))
    c = timeit(lambda l: np.vecdot(Vt[l][:, None, :, :T], att[:, :, None, :], out=out))
    print('%4d  %10.3f  %11.3f  %11.3f' % (T, a, b, c))
