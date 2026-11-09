# Instrumented copy of run.forward: accumulates time per section (min over repeats).
import sys, time
sys.path.insert(0, '.')
import numpy as np
from run import Transformer, F32
m = Transformer('model.bin')
pc = time.perf_counter
def fwd(self, token, pos, acc):
    vecdot, multiply, exp = np.vecdot, np.multiply, np.exp
    one, eps, fdim, sqrt_hs = self.one, self.eps, self.fdim, self.sqrt_hs
    x, xb, xb2, hb, hb2 = self.x, self.xb, self.xb2, self.hb, self.hb2
    q, qc, qg = self.q, self.qc, self.qg
    rot = self.rope[pos]; T = pos + 1
    x[:] = self.tok_emb[token]
    for l, (rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2) in enumerate(self.layers):
        t0 = pc()
        ss = one / np.sqrt(vecdot(x, x) / fdim + eps); multiply(x, ss, out=xb); multiply(rms_att, xb, out=xb)
        kc = self.key_cache[l]; vc = self.value_cache[l]; kslot = kc[:, pos, :]
        t1 = pc()
        vecdot(wq, xb, out=q); vecdot(wk, xb, out=kslot); vecdot(wv, xb, out=vc[:, pos, :])
        t2 = pc()
        multiply(qc, rot, out=qc); kcx = kslot.view(np.complex64); multiply(kcx, rot, out=kcx)
        t3 = pc()
        att = np.matmul(qg, kc[:, :T, :].transpose(0, 2, 1)); att /= sqrt_hs
        att -= att.max(axis=-1, keepdims=True); exp(att, out=att); att /= att.sum(axis=-1, keepdims=True, dtype=F32)
        np.matmul(att, vc[:, :T, :], out=xb.reshape(qg.shape))
        t4 = pc()
        vecdot(wo, xb, out=xb2)
        t5 = pc()
        x += xb2
        ss = one / np.sqrt(vecdot(x, x) / fdim + eps); multiply(x, ss, out=xb); multiply(rms_ffn, xb, out=xb)
        t6 = pc()
        vecdot(w1, xb, out=hb); vecdot(w3, xb, out=hb2)
        t7 = pc()
        sig = exp(np.negative(hb)); sig += one; np.divide(one, sig, out=sig); hb *= sig; hb *= hb2
        t8 = pc()
        vecdot(w2, hb, out=xb)
        t9 = pc()
        x += xb
        t10 = pc()
        for k, v in (('rmsnorm1', t1-t0), ('qkv', t2-t1), ('rope', t3-t2), ('attn', t4-t3), ('wo', t5-t4), ('res+rmsnorm2', t6-t5), ('w1w3', t7-t6), ('swiglu', t8-t7), ('w2', t9-t8), ('res', t10-t9)):
            acc[k] = acc.get(k, 0) + v
    t0 = pc()
    ss = one / np.sqrt(vecdot(x, x) / fdim + eps); multiply(x, ss, out=x); multiply(self.rms_final, x, out=x)
    t1 = pc(); vecdot(self.wcls, x, out=self.logits); t2 = pc()
    acc['final_norm'] = acc.get('final_norm', 0) + t1 - t0; acc['classifier'] = acc.get('classifier', 0) + t2 - t1
for p in range(80): m.forward(5, p)
best = None
for r in range(7):
    acc = {}
    for _ in range(30): fwd(m, 5, 80, acc)
    acc = {k: v / 30 * 1e3 for k, v in acc.items()}
    best = acc if best is None else {k: min(best[k], acc[k]) for k in acc}
tot = sum(best.values())
for k, v in sorted(best.items(), key=lambda kv: -kv[1]): print('%-14s %.3f ms' % (k, v))
print('sum %.2f ms; non-matmul %.3f ms' % (tot, sum(v for k, v in best.items() if k in ('rmsnorm1','rope','attn','res+rmsnorm2','swiglu','res','final_norm'))))
