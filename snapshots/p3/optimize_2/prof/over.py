import sys, time, importlib
sys.path.insert(0, sys.argv[1]); mod=importlib.import_module(sys.argv[2])
import numpy as np
m=mod.Transformer('model.bin')
def bench(f,n=30,r=7):
    f(); best=1e9
    for _ in range(r):
        t=time.perf_counter()
        for _ in range(n): f()
        best=min(best,(time.perf_counter()-t)/n)
    return best*1e3
x=np.ones(768,np.float32); h=np.ones(2048,np.float32)
o768=np.empty(768,np.float32); o2048=np.empty(2048,np.float32); o32k=np.empty(32000,np.float32)
W=[(m.wq[l],m.wk[l],m.wv[l],m.wo[l],m.w1[l],m.w3[l],m.w2[l]) for l in range(12)]
def mats():
    for (wq,wk,wv,wo,w1,w3,w2) in W:
        np.vecdot(wq,x,out=o768);np.vecdot(wk,x,out=o768);np.vecdot(wv,x,out=o768);np.vecdot(wo,x,out=o768)
        np.vecdot(w1,x,out=o2048);np.vecdot(w3,x,out=o2048);np.vecdot(w2,h,out=o768)
    np.vecdot(m.wcls,x,out=o32k)
m.reset()
for p in range(80): m.forward(5,p)
mt=bench(mats); ft=bench(lambda: m.forward(5,80))
print('min: matmuls %.2f ms, forward@80 %.2f ms, overhead %.2f ms'%(mt,ft,ft-mt))
