import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=100):
    for _ in range(5): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e3
L=12
q=[rng.standard_normal((768,768),dtype=np.float32) for _ in range(3*L)]
f13=[rng.standard_normal((2048,768),dtype=np.float32) for _ in range(2*L)]
qkv=[np.concatenate(q[3*l:3*l+3]) for l in range(L)]; w13=[np.concatenate(f13[2*l:2*l+2]) for l in range(L)]
x=rng.standard_normal(768,dtype=np.float32); o=np.empty(768,np.float32); o2=np.empty(2048,np.float32); o3=np.empty(2304,np.float32); o4=np.empty(4096,np.float32)
print('separate %.2f ms'%bench(lambda:([np.vecdot(W,x,out=o) for W in q],[np.vecdot(W,x,out=o2) for W in f13])))
print('fused    %.2f ms'%bench(lambda:([np.vecdot(W,x,out=o3) for W in qkv],[np.vecdot(W,x,out=o4) for W in w13])))
