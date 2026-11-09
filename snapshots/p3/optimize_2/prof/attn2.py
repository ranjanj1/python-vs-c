import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=200):
    for _ in range(5): f()
    best=1e9
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(n): f()
        best=min(best,(time.perf_counter()-t)/n)
    return best*1e6
H,hs,S=12,64,1024; F=np.float32; s=F(8.0)
kc=rng.standard_normal((H,S,hs),dtype=F); vc=rng.standard_normal((H,S,hs),dtype=F)
kt=np.ascontiguousarray(kc.transpose(0,2,1))  # (H,hs,S)
q=rng.standard_normal(H*hs,dtype=F); qg=q.reshape(H,1,hs); xb=np.empty(H*hs,F); xo=xb.reshape(H,1,hs)
def sm(att):
    att/=s; att-=att.max(axis=-1,keepdims=True); np.exp(att,out=att); att/=att.sum(axis=-1,keepdims=True,dtype=F)
for T in (80,512,1024):
    def cur():
        att=np.matmul(qg,kc[:,:T,:].transpose(0,2,1)); sm(att); np.matmul(att,vc[:,:T,:],out=xo)
    def ktl():
        att=np.matmul(qg,kt[:,:,:T]); sm(att); np.matmul(att,vc[:,:T,:],out=xo)
    def perhead():
        att=np.empty((H,1,T),F)
        for h in range(H): np.dot(kc[h,:T],q[h*hs:(h+1)*hs],out=att[h,0])
        sm(att)
        for h in range(H): np.dot(att[h,0],vc[h,:T],out=xb[h*hs:(h+1)*hs])
    def vd():
        att=np.vecdot(kc[:,:T,:],qg)[:,None,:]; sm(att); np.matmul(att,vc[:,:T,:],out=xo)
    r={k:bench(f) for k,f in dict(cur=cur,kt=ktl,perhead=perhead,vecdot=vd).items()}
    ref=None
    print(T,' '.join('%s %.1fus'%kv for kv in r.items()))
