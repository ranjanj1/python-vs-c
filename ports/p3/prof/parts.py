import sys, time, cProfile, pstats
sys.path.insert(0, sys.argv[1])
import numpy as np
from run import Transformer
m=Transformer('model.bin')
def bench(f,n=50):
    f(); t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e3
x=np.ones(768,np.float32); h=np.ones(2048,np.float32)
o768=np.empty(768,np.float32); o2048=np.empty(2048,np.float32); o32k=np.empty(32000,np.float32)
def mats():
    for l in range(12):
        np.dot(m.wq[l],x,out=o768);np.dot(m.wk[l],x,out=o768);np.dot(m.wv[l],x,out=o768);np.dot(m.wo[l],x,out=o768)
        np.dot(m.w1[l],x,out=o2048);np.dot(m.w3[l],x,out=o2048);np.dot(m.w2[l],h,out=o768)
def cls(): np.dot(m.wcls,x,out=o32k)
print('layer matmuls %.2f ms'%bench(mats)); print('classifier %.2f ms'%bench(cls))
m.reset()
for p in range(80): m.forward(5,p)
print('forward @80 %.2f ms'%bench(lambda: (m.forward(5,80))))
pr=cProfile.Profile(); pr.enable()
for i in range(50): m.forward(5,80)
pr.disable(); pstats.Stats(pr).sort_stats('tottime').print_stats(14)
