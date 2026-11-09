import time, numpy as np
F = np.float32; pc = time.perf_counter; rng = np.random.default_rng(0)
def bench(f, n=20, r=5):
    f(); best = 1e9
    for _ in range(r):
        t = pc()
        for _ in range(n): f()
        best = min(best, (pc() - t) / n)
    return best
d, n, L, b = 2048, 768, 24, 256; nb = n // b
Ws = [rng.standard_normal((d, n), dtype=F) for _ in range(L)]; gb = d * n * 4 * L / 1e9
x = rng.standard_normal(n, dtype=F); xs = x.reshape(nb, b, 1); tmp = np.empty((nb, d, 1), F); o = np.empty(d, F)
views = [W.reshape(d, nb, b).transpose(1, 0, 2) for W in Ws]
copies = [np.ascontiguousarray(v) for v in views]
print('strided view bmm %.0f GB/s' % (gb / bench(lambda: [(np.matmul(V, xs, out=tmp), np.add.reduce(tmp, axis=0, out=o[:, None])) for V in views])))
print('contig copy  bmm %.0f GB/s' % (gb / bench(lambda: [(np.matmul(V, xs, out=tmp), np.add.reduce(tmp, axis=0, out=o[:, None])) for V in copies])))
# load-time cost of reorganising the whole checkpoint into blocks
t = pc(); mm = np.memmap('model.bin', dtype=F, mode='r', offset=28); a = np.empty(mm.shape, F); 
t1 = pc(); arr = np.fromfile('model.bin', dtype=F, offset=28); t2 = pc()
blk = np.empty_like(arr); m = arr[:arr.size // 768 * 768].reshape(-1, 3, 256); blk[:m.size].reshape(3, -1, 256)[...] = m.transpose(1, 0, 2); t3 = pc()
print('fromfile %.3fs, full 768-col block reorg copy %.3fs' % (t2 - t1, t3 - t2))
