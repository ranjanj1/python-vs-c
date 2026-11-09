import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=200):
    for _ in range(5): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e6
L=12
for (d,n) in [(768,768),(2048,768),(768,2048),(2304,768),(4096,768),(32000,768)]:
    Ws=[rng.standard_normal((d,n),dtype=np.float32) for _ in range(L if d<32000 else 1)]
    x=rng.standard_normal(n,dtype=np.float32); o=np.empty(d,np.float32)
    WT=[np.ascontiguousarray(W.T) for W in Ws]  # (n,d) C-contig
    r={}
    r['dot(W,x)']=bench(lambda:[np.dot(W,x,out=o) for W in Ws])
    r['matmul(W,x)']=bench(lambda:[np.matmul(W,x,out=o) for W in Ws])
    r['dot(x,WT)']=bench(lambda:[np.dot(x,W,out=o) for W in WT])
    r['dot(x,W.T)']=bench(lambda:[np.dot(x,W.T,out=o) for W in Ws])
    gb=d*n*4*len(Ws)/1e9
    print((d,n), {k:'%.0fus %.0fGB/s'%(v, gb/(v*1e-6)) for k,v in r.items()})
