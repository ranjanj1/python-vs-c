import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from run import Transformer
m=Transformer('model.bin')
def bench(f,n=60):
    f(); t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e3
x=np.ones(768,np.float32); h=np.ones(2048,np.float32)
o768=np.empty(768,np.float32); o2048=np.empty(2048,np.float32); o32k=np.empty(32000,np.float32)
def mats():
    for (ra,wq,wk,wv,wo,rf,w1,w3,w2) in m.layers:
        np.vecdot(wq,x,out=o768);np.vecdot(wk,x,out=o768.reshape(12,64));np.vecdot(wv,x,out=o768.reshape(12,64));np.vecdot(wo,x,out=o768)
        np.vecdot(w1,x,out=o2048);np.vecdot(w3,x,out=o2048);np.vecdot(w2,h,out=o768)
    np.vecdot(m.wcls,x,out=o32k)
mt=bench(mats)
m.reset()
for p in range(80): m.forward(5,p)
ft=bench(lambda: m.forward(5,80))
print('matmuls %.2f ms, forward@80 %.2f ms, overhead %.2f ms'%(mt,ft,ft-mt))
# dtype audit
import numpy as _np
orig=_np.ndarray
x=m.x; print('ss dtype', _np.vecdot(x,x).dtype, (m.one/_np.sqrt(_np.vecdot(x,x)/m.fdim+m.eps)).dtype, m.rope.dtype, m.logits.dtype)
