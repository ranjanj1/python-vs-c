import numpy as np, time
from numba import njit, uint64

@njit
def mm_rows4(out, x, w, n, d):
    i = 0
    while i + 4 <= d:
        b0 = i*n; b1 = b0+n; b2 = b1+n; b3 = b2+n
        a0=a1=a2=a3=np.float32(0.0)
        for j in range(n):
            xj = x[j]
            a0 += w[b0+j]*xj; a1 += w[b1+j]*xj; a2 += w[b2+j]*xj; a3 += w[b3+j]*xj
        out[i]=a0; out[i+1]=a1; out[i+2]=a2; out[i+3]=a3
        i += 4

@njit
def mm_rows8(out, x, w, n, d):
    i = 0
    while i + 8 <= d:
        b0 = i*n
        a0=a1=a2=a3=a4=a5=a6=a7=np.float32(0.0)
        for j in range(n):
            xj = x[j]
            a0 += w[b0+j]*xj; a1 += w[b0+n+j]*xj; a2 += w[b0+2*n+j]*xj; a3 += w[b0+3*n+j]*xj
            a4 += w[b0+4*n+j]*xj; a5 += w[b0+5*n+j]*xj; a6 += w[b0+6*n+j]*xj; a7 += w[b0+7*n+j]*xj
        out[i]=a0; out[i+1]=a1; out[i+2]=a2; out[i+3]=a3; out[i+4]=a4; out[i+5]=a5; out[i+6]=a6; out[i+7]=a7
        i += 8

@njit
def mm_l16_u(out, x, w, n, d):
    # lanes with unsigned indexing to avoid wraparound checks
    nn = uint64(n)
    for i in range(d):
        b = uint64(i)*nn
        a0=a1=a2=a3=a4=a5=a6=a7=np.float32(0.0)
        c0=c1=c2=c3=c4=c5=c6=c7=np.float32(0.0)
        j = uint64(0)
        while j < nn:
            a0 += w[b+j]*x[j]; a1 += w[b+j+uint64(1)]*x[j+uint64(1)]; a2 += w[b+j+uint64(2)]*x[j+uint64(2)]; a3 += w[b+j+uint64(3)]*x[j+uint64(3)]
            a4 += w[b+j+uint64(4)]*x[j+uint64(4)]; a5 += w[b+j+uint64(5)]*x[j+uint64(5)]; a6 += w[b+j+uint64(6)]*x[j+uint64(6)]; a7 += w[b+j+uint64(7)]*x[j+uint64(7)]
            c0 += w[b+j+uint64(8)]*x[j+uint64(8)]; c1 += w[b+j+uint64(9)]*x[j+uint64(9)]; c2 += w[b+j+uint64(10)]*x[j+uint64(10)]; c3 += w[b+j+uint64(11)]*x[j+uint64(11)]
            c4 += w[b+j+uint64(12)]*x[j+uint64(12)]; c5 += w[b+j+uint64(13)]*x[j+uint64(13)]; c6 += w[b+j+uint64(14)]*x[j+uint64(14)]; c7 += w[b+j+uint64(15)]*x[j+uint64(15)]
            j += uint64(16)
        out[i] = (((a0+c0)+(a1+c1))+((a2+c2)+(a3+c3)))+(((a4+c4)+(a5+c5))+((a6+c6)+(a7+c7)))

@njit
def mm_acc_arr(out, x, w2, n, d):
    # 2D weights, local accumulator vector of 16 lanes
    acc = np.empty(16, np.float32)
    for i in range(d):
        for k in range(16): acc[k] = np.float32(0.0)
        for j0 in range(0, n, 16):
            for k in range(16):
                acc[k] += w2[i, j0+k]*x[j0+k]
        s = np.float32(0.0)
        for k in range(16): s += acc[k]
        out[i] = s

@njit
def mm_axpy(out, x, wt, n, d):
    # transposed weights wt (n, d): out[i] = sum_j wt[j,i]*x[j], same j order as reference
    for i in range(d): out[i] = np.float32(0.0)
    for j in range(n):
        xj = x[j]
        for i in range(d):
            out[i] += wt[j, i]*xj

for n, d in [(768, 2048), (768, 32000), (2048, 768)]:
    rng = np.random.default_rng(0)
    w = rng.standard_normal(n*d).astype(np.float32)*np.float32(0.02)
    x = rng.standard_normal(n).astype(np.float32)
    w2 = w.reshape(d, n); wt = np.ascontiguousarray(w2.T)
    ref = (w2.astype(np.float64) @ x.astype(np.float64))
    for f, W in ((mm_rows4, w), (mm_rows8, w), (mm_l16_u, w), (mm_acc_arr, w2), (mm_axpy, wt)):
        out = np.zeros(d, np.float32)
        f(out, x, W, n, d)
        t = time.perf_counter(); R = 20
        for _ in range(R): f(out, x, W, n, d)
        dt = (time.perf_counter()-t)/R
        print(f"{n}x{d} {f.__name__:12s} {dt*1e3:7.3f} ms  {n*d*4/dt/1e9:6.1f} GB/s  err {np.abs(out-ref).max():.2e}")
