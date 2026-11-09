import numpy as np, time
from numba import njit

@njit
def axpy1(out, x, wt, n, d):
    for i in range(d): out[i] = np.float32(0.0)
    for j in range(n):
        xj = x[j]
        for i in range(d):
            out[i] += wt[j, i]*xj

@njit
def axpy4(out, x, wt, n, d):
    for i in range(d): out[i] = np.float32(0.0)
    j = 0
    while j + 4 <= n:
        x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3]
        for i in range(d):
            v = out[i]
            v += wt[j, i]*x0
            v += wt[j+1, i]*x1
            v += wt[j+2, i]*x2
            v += wt[j+3, i]*x3
            out[i] = v
        j += 4
    while j < n:
        xj = x[j]
        for i in range(d):
            out[i] += wt[j, i]*xj
        j += 1

@njit
def axpy4_tiled(out, x, wt, n, d, T):
    for i0 in range(0, d, T):
        i1 = min(i0 + T, d)
        for i in range(i0, i1): out[i] = np.float32(0.0)
        j = 0
        while j + 4 <= n:
            x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3]
            for i in range(i0, i1):
                v = out[i]
                v += wt[j, i]*x0
                v += wt[j+1, i]*x1
                v += wt[j+2, i]*x2
                v += wt[j+3, i]*x3
                out[i] = v
            j += 4
        while j < n:
            xj = x[j]
            for i in range(i0, i1):
                out[i] += wt[j, i]*xj
            j += 1

@njit
def axpy8_tiled(out, x, wt, n, d, T):
    for i0 in range(0, d, T):
        i1 = min(i0 + T, d)
        for i in range(i0, i1): out[i] = np.float32(0.0)
        j = 0
        while j + 8 <= n:
            x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3]
            x4 = x[j+4]; x5 = x[j+5]; x6 = x[j+6]; x7 = x[j+7]
            for i in range(i0, i1):
                v = out[i]
                v += wt[j, i]*x0; v += wt[j+1, i]*x1; v += wt[j+2, i]*x2; v += wt[j+3, i]*x3
                v += wt[j+4, i]*x4; v += wt[j+5, i]*x5; v += wt[j+6, i]*x6; v += wt[j+7, i]*x7
                out[i] = v
            j += 8
        while j < n:
            xj = x[j]
            for i in range(i0, i1):
                out[i] += wt[j, i]*xj
            j += 1

def run(f, args, R=30):
    f(*args)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(R): f(*args)
        best = min(best, (time.perf_counter()-t)/R)
    return best

for n, d in [(768, 768), (768, 2048), (2048, 768), (768, 32000)]:
    rng = np.random.default_rng(0)
    w2 = (rng.standard_normal(n*d).astype(np.float32)*np.float32(0.02)).reshape(d, n)
    x = rng.standard_normal(n).astype(np.float32)
    wt = np.ascontiguousarray(w2.T)
    out = np.zeros(d, np.float32)
    base = None
    for name, f, extra in [("axpy1", axpy1, ()), ("axpy4", axpy4, ()),
                           ("axpy4_T512", axpy4_tiled, (512,)), ("axpy4_T1024", axpy4_tiled, (1024,)), ("axpy4_T2048", axpy4_tiled, (2048,)),
                           ("axpy8_T1024", axpy8_tiled, (1024,)), ("axpy8_T2048", axpy8_tiled, (2048,))]:
        dt = run(f, (out, x, wt, n, d) + extra)
        if base is None: base = out.copy()
        print(f"{n}x{d} {name:12s} {dt*1e3:7.3f} ms {n*d*4/dt/1e9:6.1f} GB/s same={np.array_equal(out, base)}")
