"""Separate per-matrix arrays vs one flat buffer in consumption order (contiguous views). Run one mode per process."""
import os, sys, time, statistics, json, math, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from numba import njit
import run
F32 = np.float32
mode = sys.argv[1]
m = run.Model('model.bin')
fx = json.load(open('fixtures.json')); seq = fx['prompts'][2]['seq']

@njit(cache=True, boundscheck=False)
def fwd_flat(token, pos, emb, rms_att, flat, sh, rms_ffn, rms_final, kc_all, vc_all, x, xb, xb2, qkv, h13, att, logits,
             rope_c, rope_s, n_heads, n_kv_heads, kv_dim, k_off, v_off, hidden_dim, h3_off):
    # sh: int64 rows [nt, n, T] for qkv, o, w13, w2, cls ; per-layer slab = qkv|o|w13|w2 ; cls after all layers
    T = sh[0, 2]
    s0 = sh[0, 0] * sh[0, 1] * T; s1 = sh[1, 0] * sh[1, 1] * T; s2 = sh[2, 0] * sh[2, 1] * T; s3 = sh[3, 0] * sh[3, 1] * T
    per = s0 + s1 + s2 + s3
    dim = x.shape[0]; n_layers = rms_att.shape[0]; head_size = dim // n_heads
    hs_f = F32(head_size); sqrt_hs = math.sqrt(hs_f)
    for i in range(dim): x[i] = emb[token, i]
    for i in range(0, dim, 2):
        freq = F32(1.0) / math.pow(F32(10000.0), F32(i % head_size) / hs_f)
        val = F32(pos) * freq
        rope_c[i // 2] = math.cos(val); rope_s[i // 2] = math.sin(val)
    q = qkv[:dim]; k = qkv[k_off:k_off + kv_dim]; v = qkv[v_off:v_off + kv_dim]
    hb = h13[:hidden_dim]; hb2 = h13[h3_off:h3_off + hidden_dim]
    for l in range(n_layers):
        b = l * per
        wq = flat[b:b + s0].reshape((sh[0, 0], sh[0, 1], T)); b += s0
        wo = flat[b:b + s1].reshape((sh[1, 0], sh[1, 1], T)); b += s1
        w13 = flat[b:b + s2].reshape((sh[2, 0], sh[2, 1], T)); b += s2
        w2 = flat[b:b + s3].reshape((sh[3, 0], sh[3, 1], T))
        run.rmsnorm(xb, x, rms_att[l])
        run.matvec_tiles(qkv, xb, wq)
        for i in range(0, dim, 2):
            fcr = rope_c[i // 2]; fci = rope_s[i // 2]
            v0 = q[i]; v1 = q[i + 1]; q[i] = v0 * fcr - v1 * fci; q[i + 1] = v0 * fci + v1 * fcr
            if i < kv_dim:
                v0 = k[i]; v1 = k[i + 1]; k[i] = v0 * fcr - v1 * fci; k[i + 1] = v0 * fci + v1 * fcr
        kc = kc_all[l]; vc = vc_all[l]
        for hk in range(n_kv_heads):
            for i in range(head_size): kc[hk, i, pos] = k[hk * head_size + i]
        for i in range(kv_dim): vc[pos, i] = v[i]
        run.attention(q, kc, vc, att, xb, pos, n_heads, n_kv_heads, head_size, sqrt_hs)
        run.matvec_tiles(xb2, xb, wo)
        for i in range(dim): x[i] += xb2[i]
        run.rmsnorm(xb, x, rms_ffn[l])
        run.matvec_tiles(h13, xb, w13)
        for i in range(hidden_dim):
            val = hb[i]; val *= F32(1.0) / (F32(1.0) + math.exp(-val)); val *= hb2[i]; hb[i] = val
        run.matvec_tiles(xb2, hb, w2)
        for i in range(dim): x[i] += xb2[i]
    run.rmsnorm(x, x, rms_final)
    b = n_layers * per
    run.matvec_tiles(logits, x, flat[b:b + sh[4, 0] * sh[4, 1] * T].reshape((sh[4, 0], sh[4, 1], T)))

if mode == 'flat':
    L = m.n_layers
    sh = np.array([m.wqkv.shape[1:], m.wo.shape[1:], m.w13.shape[1:], m.w2.shape[1:], m.wcls.shape], np.int64)
    per = m.wqkv[0].size + m.wo[0].size + m.w13[0].size + m.w2[0].size
    flat = np.empty(L * per + m.wcls.size, F32)
    for l in range(L):
        b = l * per
        for w in (m.wqkv[l], m.wo[l], m.w13[l], m.w2[l]):
            flat[b:b + w.size] = w.ravel(); b += w.size
    flat[L * per:] = m.wcls.ravel()
    ref = None
    del m.wqkv, m.wo, m.w13, m.w2, m.wcls; gc.collect()
    def go():
        m.reset()
        for p, t in enumerate(seq):
            fwd_flat(t, p, m.emb, m.rms_att, flat, sh, m.rms_ffn, m.rms_final, m.key_cache, m.value_cache, m.x, m.xb, m.xb2,
                     m.qkv, m.h13, m.att, m.logits_buf, m.rope_c, m.rope_s, m.n_heads, m.n_kv_heads, m.kv_dim, m.k_off, m.v_off, m.hidden_dim, m.h3_off)
        return m.logits.copy()
else:
    def go():
        m.reset()
        for p, t in enumerate(seq): m.forward(t, p)
        return m.logits.copy()
out = go()
np.save(f'/tmp/p4_flat_{mode}.npy', out)
ts = []
for _ in range(6):
    t = time.perf_counter(); go(); ts.append((time.perf_counter() - t) / len(seq))
print(f'{mode:5s} median {statistics.median(ts)*1e3:.3f} ms/token  min {min(ts)*1e3:.3f}')
