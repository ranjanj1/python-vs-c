"""tile_into + matvec_tiles on shapes that are NOT multiples of TILE (padding path) vs a float64 reference."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
F32 = np.float32
rng = np.random.default_rng(0)
for d, n in [(128, 768), (300, 77), (1000, 130), (37, 16), (5, 3)]:
    W = rng.standard_normal((2, d, n)).astype(F32)
    nt = run._ntiles(d)
    dst = np.full((2, nt + 1, n, run.TILE), np.nan, F32)
    run.tile_into(dst, 1, W)
    back = dst[:, 1:].transpose(0, 1, 3, 2).reshape(2, nt * run.TILE, n)
    assert np.array_equal(back[:, :d], W) and not np.any(back[:, d:]), (d, n)
    x = rng.standard_normal(n).astype(F32)
    out = np.zeros(nt * run.TILE, F32)
    run.matvec_tiles(out, x, dst[0, 1:])
    ref = W[0].astype(np.float64) @ x.astype(np.float64)
    seq = np.zeros(d, F32)
    for r in range(d):  # run.c order, float32
        acc = F32(0)
        for j in range(n):
            acc = F32(acc + F32(W[0, r, j] * x[j]))
        seq[r] = acc
    print(f'd={d:5d} n={n:4d}: tiles ok, padded outputs zero={not np.any(out[d:])}, '
          f'bitwise == sequential run.c-order float32 sum: {np.array_equal(out[:d], seq)}, max|err| vs f64 {np.abs(out[:d]-ref).max():.2e}')
