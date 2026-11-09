# Split in-situ attention cost (cold KV cache) into scores / softmax / weighted-sum, and try alternatives.
import sys, time
sys.path.insert(0, '.')
import numpy as np
from run import Transformer, F32
m = Transformer('model.bin'); pc = time.perf_counter
H, hs = m.n_heads, m.head_size
junk = np.ones(64 * 1024 * 1024 // 4, F32)  # 64 MB to evict caches between layers, like weight streaming
VARS = {
 'scores_matmul': lambda qg, kc, T, att: np.matmul(qg, kc[:, :T, :].transpose(0, 2, 1), out=att),
 'scores_vecdot': lambda qg, kc, T, att: np.vecdot(kc[:, :T, :], qg, out=att[:, 0, :]),
 'scores_einsum': lambda qg, kc, T, att: np.einsum('hqd,htd->hqt', qg, kc[:, :T, :], out=att),
 'scores_perhead_gemv': lambda qg, kc, T, att: [np.dot(kc[h, :T], qg[h, 0], out=att[h, 0]) for h in range(H)],
}
WVARS = {
 'wsum_matmul': lambda att, vc, T, out: np.matmul(att, vc[:, :T, :], out=out),
 'wsum_einsum': lambda att, vc, T, out: np.einsum('hqt,htd->hqd', att, vc[:, :T, :], out=out),
 'wsum_perhead_gemv': lambda att, vc, T, out: [np.dot(att[h, 0], vc[h, :T], out=out[h, 0]) for h in range(H)],
}
for p in range(1000): m.forward(5 + p % 300, p)
qg = m.qg.copy(); s = m.sqrt_hs
for T in (81, 1000):
    att = np.empty((H, 1, T), F32); out = np.empty((H, 1, hs), F32)
    def timeit(fn):
        best = 1e9
        for r in range(5):
            tot = 0
            for l in range(12):
                junk.max()  # evict
                t = pc(); fn(l); tot += pc() - t
            best = min(best, tot)
        return best * 1e3
    r = {k: timeit(lambda l, f=f: f(qg, m.key_cache[l], T, att)) for k, f in VARS.items()}
    np.matmul(qg, m.key_cache[0][:, :T, :].transpose(0, 2, 1), out=att)
    def soft(l):
        a = att.copy(); a /= s; a -= a.max(axis=-1, keepdims=True); np.exp(a, out=a); a /= a.sum(axis=-1, keepdims=True, dtype=F32)
    r['softmax(+copy)'] = timeit(soft)
    r.update({k: timeit(lambda l, f=f: f(att, m.value_cache[l], T, out)) for k, f in WVARS.items()})
    print('T=%d (12 layers, cold):' % T, ' '.join('%s %.3f' % kv for kv in r.items()))
print('--- transposed V cache (H, hs, S)')
vt = np.ascontiguousarray(m.value_cache.transpose(0, 1, 3, 2))  # (L, H, hs, S)
for T in (81, 1000):
    att = np.random.default_rng(0).random((H, 1, T), dtype=F32); out = np.empty((H, 1, hs), F32); out2 = np.empty((H, hs), F32)
    def timeit(fn):
        best = 1e9
        for r in range(5):
            tot = 0
            for l in range(12):
                junk.max(); t = pc(); fn(l); tot += pc() - t
            best = min(best, tot)
        return best * 1e3
    a = timeit(lambda l: np.matmul(att, m.value_cache[l][:, :T, :], out=out))
    b = timeit(lambda l: np.vecdot(vt[l][:, :, :T], att, out=out2))
    c = timeit(lambda l: np.matmul(vt[l][:, :, :T], att.transpose(0, 2, 1), out=out.transpose(0, 2, 1)))
    np.matmul(att, m.value_cache[5][:, :T, :], out=out); np.vecdot(vt[5][:, :, :T], att, out=out2)
    print('T=%d wsum matmul(t,d) %.3f ms | vecdot on (d,t) %.3f ms | matmul on (d,t) %.3f ms | maxdiff %.2e' % (T, a, b, c, np.abs(out[:, 0] - out2).max()))
