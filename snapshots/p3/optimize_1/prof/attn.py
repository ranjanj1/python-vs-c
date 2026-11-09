import time, numpy as np
rng=np.random.default_rng(0)
def bench(f,n=300):
    for _ in range(5): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e6
H,hs,S=12,64,1024
kc=rng.standard_normal((H,S,hs),dtype=np.float32); vc=rng.standard_normal((H,S,hs),dtype=np.float32)
q=rng.standard_normal(H*hs,dtype=np.float32); s=np.float32(8.0)
for T in (80,1024):
    def a1():
        qh=q.reshape(H,1,hs); att=np.matmul(qh,kc[:,:T,:].transpose(0,2,1)); att/=s
        att-=att.max(axis=-1,keepdims=True); np.exp(att,out=att); att/=att.sum(axis=-1,keepdims=True,dtype=np.float32)
        return np.matmul(att,vc[:,:T,:])
    def a2():
        qh=q.reshape(H,hs); att=np.vecdot(kc[:,:T,:],qh[:,None,:]); att/=s
        att-=att.max(axis=-1,keepdims=True); np.exp(att,out=att); att/=att.sum(axis=-1,keepdims=True,dtype=np.float32)
        return np.matmul(att[:,None,:],vc[:,:T,:])
    def a3():
        qh=q.reshape(H,hs,1); att=np.matmul(kc[:,:T,:],qh)[...,0]; att/=s
        att-=att.max(axis=-1,keepdims=True); np.exp(att,out=att); att/=att.sum(axis=-1,keepdims=True,dtype=np.float32)
        return np.matmul(att[:,None,:],vc[:,:T,:])
    print(T, 'a1 %.0fus a2 %.0fus a3 %.0fus'%(bench(a1),bench(a2),bench(a3)), np.abs(a1()-a2()).max(), np.abs(a1()-a3()).max())
