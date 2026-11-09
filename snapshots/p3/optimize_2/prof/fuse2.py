# Blocked gemv: separate Q,K,V and W1,W3 vs fused (concatenated along output rows), 12 layers.
import time, numpy as np
F = np.float32; pc = time.perf_counter; rng = np.random.default_rng(0)
L, D, Hd, b = 12, 768, 2048, 256; nb = D // b
def bench(f, n=20, r=7):
    f(); best = 1e9
    for _ in range(r):
        t = pc()
        for _ in range(n): f()
        best = min(best, (pc() - t) / n)
    return best * 1e3
sep = [[rng.standard_normal((nb, d, b), dtype=F) for d in (D, D, D, Hd, Hd)] for _ in range(L)]
fus = [[np.concatenate(s[:3], axis=1), np.concatenate(s[3:], axis=1)] for s in sep]
xs = rng.standard_normal((nb, b, 1), dtype=F)
pD = np.empty((nb, D, 1), F); pH = np.empty((nb, Hd, 1), F); o = np.empty(D, F); oh = np.empty(Hd, F)
p3 = np.empty((nb, 3 * D, 1), F); p2 = np.empty((nb, 2 * Hd, 1), F); o3 = np.empty(3 * D, F); o2 = np.empty(2 * Hd, F)
def s():
    for wq, wk, wv, w1, w3 in sep:
        for w in (wq, wk, wv): np.matmul(w, xs, out=pD); np.add.reduce(pD[:, :, 0], axis=0, out=o)
        for w in (w1, w3): np.matmul(w, xs, out=pH); np.add.reduce(pH[:, :, 0], axis=0, out=oh)
def f():
    for wqkv, w13 in fus:
        np.matmul(wqkv, xs, out=p3); np.add.reduce(p3[:, :, 0], axis=0, out=o3)
        np.matmul(w13, xs, out=p2); np.add.reduce(p2[:, :, 0], axis=0, out=o2)
for _ in range(2): print('separate %.3f ms   fused %.3f ms' % (bench(s), bench(f)))
