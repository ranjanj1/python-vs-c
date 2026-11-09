# Current run.py: weight matvec time alone (blocked gemv + partial-sum reduce) vs full forward, min-of-N.
import sys, time
sys.path.insert(0, '.')
import numpy as np, run
m = run.Transformer('model.bin')
def bench(f, n=30, r=7):
    f(); best = 1e9
    for _ in range(r):
        t = time.perf_counter()
        for _ in range(n): f()
        best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3
def mats():
    for (ra, wqkv, wo, rf, w13, w2) in m.layers:
        np.matmul(wqkv, m.xb_s, out=m.p_qkv); np.add.reduce(m.p_qkv[:, :, 0], axis=0, out=m.qkv)
        np.matmul(wo, m.xb_s, out=m.p_o); np.add.reduce(m.p_o[:, :, 0], axis=0, out=m.xb2)
        np.matmul(w13, m.xb_s, out=m.p_h13); np.add.reduce(m.p_h13[:, :, 0], axis=0, out=m.h13)
        np.matmul(w2, m.hb_s, out=m.p_d); np.add.reduce(m.p_d[:, :, 0], axis=0, out=m.xb2)
    np.matmul(m.wcls, m.x_s, out=m.p_cls); np.add.reduce(m.p_cls[:, :, 0], axis=0, out=m.logits)
for p in range(80): m.forward(5, p)
mt = bench(mats); ft = bench(lambda: m.forward(5, 80))
nbytes = sum(w.nbytes for l in m.layers for w in (l[1], l[2], l[4], l[5])) + m.wcls.nbytes
print('weights %.0f MB: matvecs %.2f ms (%.0f GB/s); forward@80 %.2f ms; rest (attention, norms, RoPE, Python) %.2f ms'
      % (nbytes / 1e6, mt, nbytes / mt / 1e6, ft, ft - mt))
