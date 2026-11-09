import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=100):
    for _ in range(5): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n
d,n=4096,768; L=12
Ws=[rng.standard_normal((d,n),dtype=np.float32) for _ in range(L)]
WT=[np.ascontiguousarray(W.T) for W in Ws]
gb=d*n*4*L/1e9
x=rng.standard_normal(n,dtype=np.float32); X=x[None,:]; Xc=x[:,None].copy(); o=np.empty(d,np.float32); o2=np.empty((1,d),np.float32); o3=np.empty((d,1),np.float32)
flat=[W.ravel() for W in Ws]; ones=np.ones(d*n,np.float32)
tests={
 'gemv dot(W,x)':lambda:[np.dot(W,x,out=o) for W in Ws],
 'gemm (1,n)@WT':lambda:[np.matmul(X,W,out=o2) for W in WT],
 'gemm (1,n)@W.T':lambda:[np.matmul(X,W.T,out=o2) for W in Ws],
 'gemm W@(n,1)':lambda:[np.matmul(W,Xc,out=o3) for W in Ws],
 'sdot flat':lambda:[np.dot(f,ones) for f in flat],
 'np.sum':lambda:[f.sum() for f in flat],
 'np.add.reduce max':lambda:[f.max() for f in flat],
}
for k,f in tests.items(): print('%-18s %.0f GB/s'%(k, gb/bench(f)))
print('--- more')
o_=np.empty(d,np.float32)
more={
 'einsum ij,j':lambda:[np.einsum('ij,j->i',W,x,out=o_) for W in Ws],
 'einsum optimize':lambda:[np.einsum('ij,j->i',W,x,out=o_,optimize=True) for W in Ws],
 'vecdot':lambda:[np.vecdot(W,x,out=o_) for W in Ws],
 'matvec':lambda:[np.matvec(W,x,out=o_) for W in Ws],
 'einsum WT j,ji':lambda:[np.einsum('j,ji->i',x,W,out=o_) for W in WT],
}
for k,f in more.items():
    try: print('%-18s %.0f GB/s'%(k, gb/bench(f,30)))
    except Exception as e: print(k,'ERR',e)
r=np.einsum('ij,j->i',Ws[0],x); print('einsum err', np.abs(r-Ws[0]@x).max())
