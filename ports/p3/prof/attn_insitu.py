# Time attention variants inside the real forward loop (cold KV cache after weight streaming).
import sys, time
sys.path.insert(0, '.')
import numpy as np
from run import Transformer, F32
m = Transformer('model.bin'); pc = time.perf_counter
H, hs = m.n_heads, m.head_size
def make(var):
    def attn(qg, kc, vc, T, xb, sqrt_hs):
        if var == 'cur':
            att = np.matmul(qg, kc[:, :T, :].transpose(0, 2, 1))
        elif var == 'vecdot':
            att = np.vecdot(kc[:, :T, :], qg)[:, None, :]
        elif var == 'einsum':
            att = np.einsum('hqd,htd->hqt', qg, kc[:, :T, :])
        att /= sqrt_hs; att -= att.max(axis=-1, keepdims=True); np.exp(att, out=att)
        att /= att.sum(axis=-1, keepdims=True, dtype=F32)
        if var == 'einsum':
            np.einsum('hqt,htd->hqd', att, vc[:, :T, :], out=xb.reshape(qg.shape))
        else:
            np.matmul(att, vc[:, :T, :], out=xb.reshape(qg.shape))
    return attn
def fwd(self, token, pos, attn):
    vecdot, multiply, exp = np.vecdot, np.multiply, np.exp
    one, eps, fdim, sqrt_hs = self.one, self.eps, self.fdim, self.sqrt_hs
    x, xb, xb2, hb, hb2 = self.x, self.xb, self.xb2, self.hb, self.hb2
    q, qc, qg = self.q, self.qc, self.qg; rot = self.rope[pos]; T = pos + 1; tA = 0
    x[:] = self.tok_emb[token]
    for l, (rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2) in enumerate(self.layers):
        ss = one / np.sqrt(vecdot(x, x) / fdim + eps); multiply(x, ss, out=xb); multiply(rms_att, xb, out=xb)
        kc = self.key_cache[l]; vc = self.value_cache[l]; kslot = kc[:, pos, :]
        vecdot(wq, xb, out=q); vecdot(wk, xb, out=kslot); vecdot(wv, xb, out=vc[:, pos, :])
        multiply(qc, rot, out=qc); kcx = kslot.view(np.complex64); multiply(kcx, rot, out=kcx)
        t0 = pc(); attn(qg, kc, vc, T, xb, sqrt_hs); tA += pc() - t0
        vecdot(wo, xb, out=xb2); x += xb2
        ss = one / np.sqrt(vecdot(x, x) / fdim + eps); multiply(x, ss, out=xb); multiply(rms_ffn, xb, out=xb)
        vecdot(w1, xb, out=hb); vecdot(w3, xb, out=hb2)
        sig = exp(np.negative(hb)); sig += one; np.divide(one, sig, out=sig); hb *= sig; hb *= hb2
        vecdot(w2, hb, out=xb); x += xb
    return tA
for p in range(1000): m.forward(5 + p % 300, p)
for pos in (80, 999):
    res = {}
    for var in ('cur', 'vecdot', 'einsum'):
        a = make(var); best = 1e9
        for r in range(5):
            s = sum(fwd(m, 5, pos, a) for _ in range(20)) / 20; best = min(best, s)
        res[var] = best * 1e3
    # numerical agreement of the attention output for one layer
    print('pos', pos, ' '.join('%s %.3f ms' % kv for kv in res.items()))
qg = m.qg.copy(); kc = m.key_cache[3]; vc = m.value_cache[3]; outs = {}
for var in ('cur', 'vecdot', 'einsum'):
    xb = np.empty(768, F32); make(var)(qg, kc, vc, 900, xb, m.sqrt_hs); outs[var] = xb
print('max diff vs cur:', {k: float(np.abs(v - outs['cur']).max()) for k, v in outs.items()})
