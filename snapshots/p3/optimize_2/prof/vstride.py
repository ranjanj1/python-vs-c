# Weighted-sum cost (12 layers, cold cache) vs context length for V layouts / row padding.
import time, numpy as np
F = np.float32; pc = time.perf_counter; rng = np.random.default_rng(0)
L, H, hs, S = 12, 12, 64, 1024
junk = np.ones(64 * 1024 * 1024 // 4, F)
V = rng.standard_normal((L, H, S, hs), dtype=F)
layouts = {'matmul(t,d)': None}
for pad in (0, 16, 32, 64):
    vt = np.zeros((L, H, hs, S + pad), F); vt[..., :S] = V.transpose(0, 1, 3, 2); layouts['vecdot(d,t) pad%d' % pad] = vt
def timeit(fn):
    best = 1e9
    for r in range(7):
        tot = 0
        for l in range(L):
            junk.max(); t = pc(); fn(l); tot += pc() - t
        best = min(best, tot)
    return best * 1e3
for T in (100, 300, 600, 800, 999):
    att = rng.random((H, 1, T), dtype=F); out = np.empty((H, 1, hs), F)
    res = {}
    for k, vt in layouts.items():
        if vt is None: res[k] = timeit(lambda l: np.matmul(att, V[l][:, :T, :], out=out))
        else: res[k] = timeit(lambda l, vt=vt: np.vecdot(vt[l][:, None, :, :T], att[:, :, None, :], out=out))
    print('T=%4d ' % T + ' '.join('%s %.3f' % kv for kv in res.items()))
