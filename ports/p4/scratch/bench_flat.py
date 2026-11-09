"""Do weights laid out in one buffer, in consumption order (qkv,o,w13,w2 per layer, then cls), stream faster?"""
import os, sys, time, statistics, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from numba import njit
import run
F32 = np.float32
m = run.Model('model.bin')
fx = json.load(open('fixtures.json')); seq = fx['prompts'][2]['seq']

@njit(boundscheck=False)
def fwd_layers(token, pos, emb, rms_att, Lqkv, Lo, L13, L2, rms_ffn, rms_final, wcls, kc, vc, x, xb, xb2, qkv, h13, att, logits, rc, rs, H, KV, kv_dim, k_off, v_off, hid, h3_off):
    # same as run.forward but takes per-layer weights from tuples of views (one flat buffer)
    run.forward(token, pos, emb, rms_att, Lqkv, Lo, rms_ffn, L13, L2, rms_final, wcls, kc, vc, x, xb, xb2, qkv, h13, att, logits, rc, rs, H, KV, kv_dim, k_off, v_off, hid, h3_off)

L = m.n_layers
sizes = [m.wqkv[0].size, m.wo[0].size, m.w13[0].size, m.w2[0].size]
per = sum(sizes)
flat = np.empty(L * per + m.wcls.size, F32)
# layer-major consumption order: [qkv_l, o_l, w13_l, w2_l] for l in 0..L-1, then classifier
blk = flat[:L * per].reshape(L, per)
offs = np.cumsum([0] + sizes)
def view(k, arr):
    return blk[:, offs[k]:offs[k + 1]]  # (L, size) strided view; copy then expose as (L, ...) needs contiguity per layer
for l in range(L):
    blk[l, offs[0]:offs[1]] = m.wqkv[l].ravel(); blk[l, offs[1]:offs[2]] = m.wo[l].ravel()
    blk[l, offs[2]:offs[3]] = m.w13[l].ravel(); blk[l, offs[3]:offs[4]] = m.w2[l].ravel()
flat[L * per:] = m.wcls.ravel()
# numba needs (L, tiles, n, T) arrays; per-layer slabs are interleaved, so build strided 4D views via as_strided
from numpy.lib.stride_tricks import as_strided
def strided(k, shape):
    base = blk[:, offs[k]:]
    return as_strided(base, shape=(L,) + shape, strides=(per * 4,) + tuple(s for s in np.empty(shape, F32).strides))
Vqkv = strided(0, m.wqkv.shape[1:]); Vo = strided(1, m.wo.shape[1:]); V13 = strided(2, m.w13.shape[1:]); V2 = strided(3, m.w2.shape[1:])
Vcls = flat[L * per:].reshape(m.wcls.shape)
assert np.array_equal(Vqkv, m.wqkv) and np.array_equal(V2, m.w2) and np.array_equal(Vcls, m.wcls)
print('layouts: per-layer arrays', m.wqkv.flags['C_CONTIGUOUS'], ' flat views C-contig:', Vqkv.flags['C_CONTIGUOUS'], '(inner per-layer slabs are contiguous)')

def run_cur():
    m.reset()
    for p, t in enumerate(seq): m.forward(t, p)
    return m.logits.copy()
def run_flat():
    m.reset()
    for p, t in enumerate(seq):
        run.forward(t, p, m.emb, m.rms_att, Vqkv, Vo, m.rms_ffn, V13, V2, m.rms_final, Vcls, m.key_cache, m.value_cache,
                    m.x, m.xb, m.xb2, m.qkv, m.h13, m.att, m.logits_buf, m.rope_c, m.rope_s, m.n_heads, m.n_kv_heads,
                    m.kv_dim, m.k_off, m.v_off, m.hidden_dim, m.h3_off)
    return m.logits.copy()
a = run_cur(); b = run_flat(); print('identical:', np.array_equal(a, b))
R = {'separate arrays': [], 'one flat buffer': []}
for _ in range(8):
    for k, f in (('separate arrays', run_cur), ('one flat buffer', run_flat)):
        t = time.perf_counter(); f(); R[k].append((time.perf_counter() - t) / len(seq))
for k, v in R.items(): print(f'{k:16s} median {statistics.median(v)*1e3:.3f} ms/token  min {min(v)*1e3:.3f}')
