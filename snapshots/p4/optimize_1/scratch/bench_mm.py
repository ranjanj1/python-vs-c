import numpy as np, time
from numba import njit

@njit(cache=False)
def mm_naive(out, x, w, n, d):
    for i in range(d):
        v = np.float32(0.0)
        for j in range(n):
            v += w[i*n+j]*x[j]
        out[i] = v

@njit(cache=False)
def mm_lanes8(out, x, w, n, d):
    for i in range(d):
        b = i*n
        a0=a1=a2=a3=a4=a5=a6=a7=np.float32(0.0)
        for j in range(0, n, 8):
            a0 += w[b+j]*x[j]; a1 += w[b+j+1]*x[j+1]; a2 += w[b+j+2]*x[j+2]; a3 += w[b+j+3]*x[j+3]
            a4 += w[b+j+4]*x[j+4]; a5 += w[b+j+5]*x[j+5]; a6 += w[b+j+6]*x[j+6]; a7 += w[b+j+7]*x[j+7]
        out[i] = ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))

@njit(cache=False)
def mm_lanes16(out, x, w, n, d):
    for i in range(d):
        b = i*n
        a0=a1=a2=a3=a4=a5=a6=a7=np.float32(0.0)
        c0=c1=c2=c3=c4=c5=c6=c7=np.float32(0.0)
        for j in range(0, n, 16):
            a0 += w[b+j]*x[j]; a1 += w[b+j+1]*x[j+1]; a2 += w[b+j+2]*x[j+2]; a3 += w[b+j+3]*x[j+3]
            a4 += w[b+j+4]*x[j+4]; a5 += w[b+j+5]*x[j+5]; a6 += w[b+j+6]*x[j+6]; a7 += w[b+j+7]*x[j+7]
            c0 += w[b+j+8]*x[j+8]; c1 += w[b+j+9]*x[j+9]; c2 += w[b+j+10]*x[j+10]; c3 += w[b+j+11]*x[j+11]
            c4 += w[b+j+12]*x[j+12]; c5 += w[b+j+13]*x[j+13]; c6 += w[b+j+14]*x[j+14]; c7 += w[b+j+15]*x[j+15]
        out[i] = (((a0+c0)+(a1+c1))+((a2+c2)+(a3+c3)))+(((a4+c4)+(a5+c5))+((a6+c6)+(a7+c7)))

@njit(cache=False)
def mm_rows4(out, x, w, n, d):
    # 4 rows at a time, sequential j order per row (same order as reference)
    i = 0
    while i + 4 <= d:
        b0 = i*n; b1 = b0+n; b2 = b1+n; b3 = b2+n
        a0=a1=a2=a3=np.float32(0.0)
        for j in range(n):
            xj = x[j]
            a0 += w[b0+j]*xj; a1 += w[b1+j]*xj; a2 += w[b2+j]*xj; a3 += w[b3+j]*xj
        out[i]=a0; out[i+1]=a1; out[i+2]=a2; out[i+3]=a3
        i += 4
    while i < d:
        v = np.float32(0.0)
        for j in range(n):
            v += w[i*n+j]*x[j]
        out[i] = v; i += 1

@njit(cache=False)
def mm_r4l8(out, x, w, n, d):
    i = 0
    while i + 4 <= d:
        b0 = i*n; b1 = b0+n; b2 = b1+n; b3 = b2+n
        p0=p1=p2=p3=p4=p5=p6=p7=np.float32(0.0)
        q0=q1=q2=q3=q4=q5=q6=q7=np.float32(0.0)
        r0=r1=r2=r3=r4=r5=r6=r7=np.float32(0.0)
        s0=s1=s2=s3=s4=s5=s6=s7=np.float32(0.0)
        for j in range(0, n, 8):
            x0=x[j];x1=x[j+1];x2=x[j+2];x3=x[j+3];x4=x[j+4];x5=x[j+5];x6=x[j+6];x7=x[j+7]
            p0+=w[b0+j]*x0;p1+=w[b0+j+1]*x1;p2+=w[b0+j+2]*x2;p3+=w[b0+j+3]*x3;p4+=w[b0+j+4]*x4;p5+=w[b0+j+5]*x5;p6+=w[b0+j+6]*x6;p7+=w[b0+j+7]*x7
            q0+=w[b1+j]*x0;q1+=w[b1+j+1]*x1;q2+=w[b1+j+2]*x2;q3+=w[b1+j+3]*x3;q4+=w[b1+j+4]*x4;q5+=w[b1+j+5]*x5;q6+=w[b1+j+6]*x6;q7+=w[b1+j+7]*x7
            r0+=w[b2+j]*x0;r1+=w[b2+j+1]*x1;r2+=w[b2+j+2]*x2;r3+=w[b2+j+3]*x3;r4+=w[b2+j+4]*x4;r5+=w[b2+j+5]*x5;r6+=w[b2+j+6]*x6;r7+=w[b2+j+7]*x7
            s0+=w[b3+j]*x0;s1+=w[b3+j+1]*x1;s2+=w[b3+j+2]*x2;s3+=w[b3+j+3]*x3;s4+=w[b3+j+4]*x4;s5+=w[b3+j+5]*x5;s6+=w[b3+j+6]*x6;s7+=w[b3+j+7]*x7
        out[i]=((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))
        out[i+1]=((q0+q1)+(q2+q3))+((q4+q5)+(q6+q7))
        out[i+2]=((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
        out[i+3]=((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7))
        i += 4

for n, d in [(768, 2048), (768, 32000), (2048, 768)]:
    rng = np.random.default_rng(0)
    w = rng.standard_normal(n*d).astype(np.float32)*np.float32(0.02)
    x = rng.standard_normal(n).astype(np.float32)
    ref = (w.reshape(d, n).astype(np.float64) @ x.astype(np.float64))
    for f in (mm_naive, mm_lanes8, mm_lanes16, mm_rows4, mm_r4l8):
        out = np.zeros(d, np.float32)
        f(out, x, w, n, d)
        t = time.perf_counter(); R = 20
        for _ in range(R): f(out, x, w, n, d)
        dt = (time.perf_counter()-t)/R
        print(f"{n}x{d} {f.__name__:12s} {dt*1e3:7.3f} ms  {n*d*4/dt/1e9:6.1f} GB/s  err {np.abs(out-ref).max():.2e}")
