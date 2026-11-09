"""Profile run.py on the supplied development workloads (fixtures.json prompts, as check.py runs them).

Usage: python3 scratch/profile.py [e2e|phases|all] [repeats]
  e2e    : drive `python3 run.py` over the protocol exactly like check.py --tokens 64 --prompts 3,
           timing every command (median over repeats).
  phases : per-phase breakdown of one forward pass, calling run.py's compiled kernels phase by phase
           (same kernels, Python-driven so each phase can be timed).
"""
import json, os, subprocess, sys, tempfile, time, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
fx = json.load(open('fixtures.json'))


def e2e(repeats):
    rows = {}
    for r in range(repeats):
        t0 = time.perf_counter()
        p = subprocess.Popen([sys.executable, os.environ.get('PROG', 'run.py'), 'model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, bufsize=1)
        rd = lambda: p.stdout.readline().rstrip('\n')
        assert rd() == 'READY'
        rows.setdefault('startup', []).append(time.perf_counter() - t0)
        t = time.perf_counter()
        for text in fx['tokenizer']:
            p.stdin.write(f'ENCODE {text}\n'); p.stdin.flush(); rd()
        rows.setdefault('encode x9', []).append(time.perf_counter() - t)
        for i, pr in enumerate(fx['prompts']):
            t = time.perf_counter(); first = None
            p.stdin.write(f"GENERATE 64 {pr['text']}\n"); p.stdin.flush(); n = 0
            while True:
                s = rd()
                if first is None: first = time.perf_counter() - t
                if s.startswith('T '): n += 1
                else: break
            dt = time.perf_counter() - t
            rows.setdefault(f'gen{i} total', []).append(dt)
            rows.setdefault(f'gen{i} first-token', []).append(first)
            rows.setdefault(f'gen{i} tok/s', []).append(n / dt)
            with tempfile.TemporaryDirectory() as td:
                seq = pr['seq']
                t = time.perf_counter()
                p.stdin.write(f"LOGITS {td}/l.f32 " + ' '.join(map(str, seq)) + '\n'); p.stdin.flush(); rd()
                dt = time.perf_counter() - t
            rows.setdefault(f'logits{i} ({len(seq)} pos) total', []).append(dt)
            rows.setdefault(f'logits{i} pos/s', []).append(len(seq) / dt)
        p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
        rows.setdefault('whole session', []).append(time.perf_counter() - t0)
    print(f'{"metric":32s} {"median":>9s} {"min":>9s} {"max":>9s}')
    for k, v in rows.items():
        print(f'{k:32s} {statistics.median(v):9.3f} {min(v):9.3f} {max(v):9.3f}')


def phases(repeats):
    """Per-phase breakdown using run.py's own compiled kernels, called phase by phase."""
    import math
    from numba import njit
    import run
    F32 = np.float32
    m = run.Model('model.bin')
    hs = m.dim // m.n_heads; sqrt_hs = math.sqrt(F32(hs))
    q = m.qkv[:m.dim]; k = m.qkv[m.k_off:m.k_off + m.kv_dim]; v = m.qkv[m.v_off:m.v_off + m.kv_dim]
    hb = m.h13[:m.hidden_dim]; hb2 = m.h13[m.h3_off:m.h3_off + m.hidden_dim]

    @njit(boundscheck=False)
    def rope_coef(c, s_, pos, head_size):
        hs_f = F32(head_size)
        for i in range(0, 2 * c.shape[0], 2):
            freq = F32(1.0) / math.pow(F32(10000.0), F32(i % head_size) / hs_f)
            val = F32(pos) * freq
            c[i // 2] = math.cos(val); s_[i // 2] = math.sin(val)

    @njit(boundscheck=False)
    def rope_apply(q, k, c, s_, kv_dim):
        for i in range(0, q.shape[0], 2):
            fcr = c[i // 2]; fci = s_[i // 2]
            v0 = q[i]; v1 = q[i + 1]; q[i] = v0 * fcr - v1 * fci; q[i + 1] = v0 * fci + v1 * fcr
            if i < kv_dim:
                v0 = k[i]; v1 = k[i + 1]; k[i] = v0 * fcr - v1 * fci; k[i + 1] = v0 * fci + v1 * fcr

    @njit(boundscheck=False)
    def kv_store(k, v, kc, vc, pos, n_kv_heads, head_size):
        for hk in range(n_kv_heads):
            blk = kc[hk, pos // kc.shape[3]]
            for i in range(head_size):
                blk[i, pos % kc.shape[3]] = k[hk * head_size + i]
        for i in range(v.shape[0]):
            vc[pos, i] = v[i]

    @njit(boundscheck=False)
    def resid(x, y):
        for i in range(x.shape[0]): x[i] += y[i]

    @njit(boundscheck=False)
    def swiglu(hb, hb2):
        for i in range(hb.shape[0]):
            val = hb[i]; val *= F32(1.0) / (F32(1.0) + math.exp(-val)); val *= hb2[i]; hb[i] = val

    T = {}
    def tm(name, f, *a):
        t = time.perf_counter(); f(*a); T[name] = T.get(name, 0.0) + time.perf_counter() - t

    def fwd(token, pos):
        tm('embed', run.embed, m.x, m.emb, token)
        tm('rope coef', rope_coef, m.rope_c, m.rope_s, pos, hs)
        for l in range(m.n_layers):
            tm('rmsnorm', run.rmsnorm, m.xb, m.x, m.rms_att[l])
            tm('matvec qkv', run.matvec_tiles, m.qkv, m.xb, m.wqkv[l])
            tm('rope apply', rope_apply, q, k, m.rope_c, m.rope_s, m.kv_dim)
            tm('kv store', kv_store, k, v, m.key_cache[l], m.value_cache[l], pos, m.n_kv_heads, hs)
            tm('attention', run.attention, q, m.key_cache[l], m.value_cache[l], m.att, m.xb, pos, m.n_heads, m.n_kv_heads, hs, sqrt_hs)
            tm('matvec o', run.matvec_tiles, m.xb2, m.xb, m.wo[l])
            tm('residual', resid, m.x, m.xb2[:m.dim])
            tm('rmsnorm', run.rmsnorm, m.xb, m.x, m.rms_ffn[l])
            tm('matvec w1|w3', run.matvec_tiles, m.h13, m.xb, m.w13[l])
            tm('swiglu', swiglu, hb, hb2)
            tm('matvec w2', run.matvec_tiles, m.xb2, hb, m.w2[l])
            tm('residual', resid, m.x, m.xb2[:m.dim])
        tm('rmsnorm', run.rmsnorm, m.x, m.x, m.rms_final)
        tm('classifier', run.matvec_tiles, m.logits_buf, m.x, m.wcls)

    seq = fx['prompts'][2]['seq']
    fwd(seq[0], 0); T.clear()
    for _ in range(repeats):
        m.reset()
        for pos, tok in enumerate(seq):
            fwd(tok, pos)
    n = repeats * len(seq)
    tf = []
    for _ in range(repeats):
        m.reset(); t = time.perf_counter()
        for pos, tok in enumerate(seq):
            m.forward(tok, pos)
        tf.append((time.perf_counter() - t) / len(seq))
    tot = sum(T.values()) / n
    sizes = {'matvec qkv': m.wqkv.nbytes, 'matvec o': m.wo.nbytes, 'matvec w1|w3': m.w13.nbytes,
             'matvec w2': m.w2.nbytes, 'classifier': m.wcls.nbytes}
    print(f'dev sequence: prompt 3, {len(seq)} positions x {repeats}')
    print(f'{"phase":14s} {"ms/token":>9s} {"share":>6s}  GB/s')
    for kk, vv in sorted(T.items(), key=lambda kv: -kv[1]):
        bw = f'{sizes[kk] / (vv / n) / 1e9:6.1f}' if kk in sizes else ''
        print(f'{kk:14s} {vv / n * 1e3:9.3f} {100 * vv / n / tot:5.1f}%  {bw}')
    print(f'{"sum (phased)":14s} {tot*1e3:9.3f}')
    print(f'compiled forward(): median {statistics.median(tf)*1e3:.3f} ms/token, min {min(tf)*1e3:.3f}')
    wbytes = sum(sizes.values())
    print(f'weights streamed per token {wbytes/1e6:.0f} MB -> effective {wbytes/min(tf)/1e9:.1f} GB/s at min')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if mode in ('phases', 'all'):
        phases(reps)
    if mode in ('e2e', 'all'):
        e2e(reps)
