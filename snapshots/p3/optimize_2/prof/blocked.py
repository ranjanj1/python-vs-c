# Column-blocked weights: W (d,n) stored as (n/b, d, b) contiguous blocks; y = sum_k Wk @ x[k*b:(k+1)*b].
import time, numpy as np
F = np.float32; pc = time.perf_counter; rng = np.random.default_rng(0)
def bench(f, n=20, r=5):
    for _ in range(2): f()
    best = 1e9
    for _ in range(r):
        t = pc()
        for _ in range(n): f()
        best = min(best, (pc() - t) / n)
    return best
for (d, n, L) in [(768, 768, 48), (2048, 768, 24), (768, 2048, 12), (32000, 768, 1)]:
    Ws = [rng.standard_normal((d, n), dtype=F) for _ in range(L)]
    x = rng.standard_normal(n, dtype=F); o = np.empty(d, F); gb = d * n * 4 * L / 1e9
    res = {'vecdot': bench(lambda: [np.vecdot(W, x, out=o) for W in Ws])}
    for b in (128, 192, 256, 384):
        if n % b: continue
        nb = n // b
        Bs = [np.ascontiguousarray(W.reshape(d, nb, b).transpose(1, 0, 2)) for W in Ws]  # (nb, d, b)
        xs = x.reshape(nb, b); tmp = np.empty((nb, d), F)
        def blk(Bs=Bs, xs=xs, tmp=tmp, nb=nb):
            for B in Bs:
                for k in range(nb): np.dot(B[k], xs[k], out=tmp[k])
                np.add.reduce(tmp, axis=0, out=o)
        def blk_vd(Bs=Bs, xs=xs, tmp=tmp):
            for B in Bs:
                np.matmul(B, xs[:, :, None], out=tmp[:, :, None]); np.add.reduce(tmp, axis=0, out=o)
        res['gemv b=%d' % b] = bench(blk)
        res['bmm b=%d' % b] = bench(blk_vd)
        Bs[0] is not None
        y = None
        for k in range(nb): np.dot(Bs[0][k], xs[k], out=tmp[k])
        err = np.abs(tmp.sum(0, dtype=F) - Ws[0].astype(np.float64) @ x.astype(np.float64)).max()
        res['err b=%d' % b] = err
    print((d, n), ' '.join(('%s %.0fGB/s' % (k, gb / v)) if not k.startswith('err') else '%s %.1e' % (k, v) for k, v in res.items()))
