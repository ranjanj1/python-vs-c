import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from numba import njit
F32 = np.float32
@njit(boundscheck=False)
def tile_b(dst, t0, src):
    # full tiles: 16x16 blocks, no per-element branch; padded tail tile handled separately
    L, d, n = src.shape
    T = dst.shape[3]
    nfull = d // T
    for l in range(L):
        s = src[l]
        for t in range(nfull):
            blk = dst[l, t0 + t]
            r0 = t * T
            for c0 in range(0, T, 16):
                for j0 in range(0, n - 15, 16):
                    for c in range(c0, c0 + 16):
                        sr = s[r0 + c]
                        for j in range(j0, j0 + 16):
                            blk[j, c] = sr[j]
                for j in range((n // 16) * 16, n):
                    for c in range(c0, c0 + 16):
                        blk[j, c] = s[r0 + c, j]
        if nfull * T < d or True:
            pass
    return nfull
@njit(boundscheck=False)
def tile_c(dst, t0, src):
    # read rows j-blocked, write each tile row (128 contiguous floats) per j
    L, d, n = src.shape
    T = dst.shape[3]
    nfull = d // T
    for l in range(L):
        s = src[l]
        for t in range(nfull):
            blk = dst[l, t0 + t]
            r0 = t * T
            for j0 in range(0, n, 32):
                j1 = min(j0 + 32, n)
                for c in range(T):
                    sr = s[r0 + c]
                    for j in range(j0, j1):
                        blk[j, c] = sr[j]
    return nfull
L, d, n = 12, 4096, 768
src = np.random.default_rng(0).standard_normal((L, d, n)).astype(F32)
for name, f in (('current tile_into', run.tile_into), ('tile_b 16x16', tile_b), ('tile_c j32', tile_c)):
    dst = np.empty((L, d // 128, n, 128), F32)
    f(dst, 0, src[:1]); best = 1e9
    for _ in range(3):
        dst = np.empty((L, d // 128, n, 128), F32)
        t = time.perf_counter(); f(dst, 0, src); best = min(best, time.perf_counter() - t)
    ok = np.array_equal(dst.transpose(0, 1, 3, 2).reshape(L, d, n), src)
    print(f'{name:18s} {best*1e3:7.1f} ms for {src.nbytes/1e6:.0f} MB ({src.nbytes/best/1e9:.1f} GB/s) correct={ok}')
