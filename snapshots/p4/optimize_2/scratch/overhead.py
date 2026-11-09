"""Per-call overheads outside the forward pass: dispatch of forward() with its 29 args, argmax, reset, LOGITS copy/write."""
import os, sys, time, statistics, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from numba import njit
import run
m = run.Model('model.bin'); m.forward(1, 0); run.argmax(m.logits)
@njit(cache=False)
def nop29(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28):
    return a0
args = (1, 0, m.emb, m.rms_att, m.wqkv, m.wo, m.rms_ffn, m.w13, m.w2, m.rms_final, m.wcls, m.key_cache, m.value_cache,
        m.x, m.xb, m.xb2, m.qkv, m.h13, m.att, m.logits_buf, m.rope_c, m.rope_s, m.n_heads, m.n_kv_heads,
        m.kv_dim, m.k_off, m.v_off, m.hidden_dim, m.h3_off)
nop29(*args)
def bench(f, R=2000):
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(R): f()
        ts.append((time.perf_counter() - t) / R)
    return min(ts) * 1e6
print(f'dispatch of a 29-arg njit call with the same arguments: {bench(lambda: nop29(*args)):.1f} us')
print(f'Model.forward python wrapper + dispatch (nop body):    {bench(lambda: nop29(*(1, 0, m.emb, m.rms_att, m.wqkv, m.wo, m.rms_ffn, m.w13, m.w2, m.rms_final, m.wcls, m.key_cache, m.value_cache, m.x, m.xb, m.xb2, m.qkv, m.h13, m.att, m.logits_buf, m.rope_c, m.rope_s, m.n_heads, m.n_kv_heads, m.kv_dim, m.k_off, m.v_off, m.hidden_dim, m.h3_off))):.1f} us')
print(f'argmax over 32000:                                     {bench(lambda: run.argmax(m.logits), 500):.1f} us')
print(f'reset (zero 2x{m.key_cache.nbytes/1e6:.0f} MB KV cache):                    {bench(lambda: m.reset(), 50):.0f} us')
allv = np.empty((82, m.vocab), np.float32)
def cp():
    for p in range(82): allv[p] = m.logits
print(f'LOGITS: 82 row copies:                                 {bench(cp, 20):.0f} us')
with tempfile.TemporaryDirectory() as td:
    print(f'LOGITS: astype+tofile 82x32000:                        {bench(lambda: allv.astype("<f4").tofile(td + "/l.f32"), 10):.0f} us')
    print(f'LOGITS: tofile only:                                   {bench(lambda: allv.tofile(td + "/l.f32"), 10):.0f} us')
