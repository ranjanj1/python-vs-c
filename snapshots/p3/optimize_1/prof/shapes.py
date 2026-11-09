import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=60):
    for _ in range(3): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n
for (d,n,L) in [(768,768,48),(2048,768,24),(768,2048,12),(32000,768,1)]:
    Ws=[rng.standard_normal((d,n),dtype=np.float32) for _ in range(L)]
    WT=[np.ascontiguousarray(W.T) for W in Ws]
    x=rng.standard_normal(n,dtype=np.float32); o=np.empty(d,np.float32)
    gb=d*n*4*L/1e9
    r={'gemv':lambda:[np.dot(W,x,out=o) for W in Ws],
       'vecdot':lambda:[np.vecdot(W,x,out=o) for W in Ws],
       'einsumT':lambda:[np.einsum('j,ji->i',x,W,out=o) for W in WT]}
    print((d,n,L), ' '.join('%s %.0fGB/s'%(k,gb/bench(f)) for k,f in r.items()))
    W=Ws[0].astype(np.float64); ref=W@x.astype(np.float64)
    print('   errs', 'gemv %.2e'%np.abs(np.dot(Ws[0],x)-ref).max(), 'vecdot %.2e'%np.abs(np.vecdot(Ws[0],x)-ref).max(), 'einsumT %.2e'%np.abs(np.einsum('j,ji->i',x,WT[0])-ref).max())
