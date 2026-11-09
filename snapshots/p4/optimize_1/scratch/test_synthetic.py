"""Synthetic checkpoint with awkward shapes (padding, GQA, unshared classifier): new forward vs v1, bitwise."""
import os, sys, struct, tempfile, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
old = load('run_v1', os.path.join(ROOT, 'scratch', 'run_v1.py')); new = load('run', os.path.join(ROOT, 'run.py'))
rng = np.random.default_rng(7)
for (dim, hid, L, H, KV, V, S, shared) in [(200, 300, 2, 5, 1, 1000, 40, False), (256, 640, 2, 8, 4, 515, 33, True), (768, 2048, 1, 12, 12, 32000, 16, True)]:
    hs = dim // H; kv = dim * KV // H
    parts = [V * dim, L * dim, L * dim * dim, L * dim * kv, L * dim * kv, L * dim * dim, L * dim,
             L * dim * hid, L * hid * dim, L * dim * hid, dim, S * hs // 2, S * hs // 2] + ([] if shared else [V * dim])
    w = (rng.standard_normal(sum(parts)) * 0.1).astype('<f4')
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(struct.pack('<7i', dim, hid, L, H, KV, V if shared else -V, S)); f.write(w.tobytes()); p = f.name
    mo, mn = old.Model(p), new.Model(p)
    ok = True; md = 0.0
    for m in (mo, mn): m.reset()
    toks = rng.integers(0, V, S)
    for pos in range(S):
        a = mo.forward(int(toks[pos]), pos).copy(); b = mn.forward(int(toks[pos]), pos).copy()
        ok &= np.array_equal(a, b); md = max(md, float(np.abs(a - b).max()))
    os.unlink(p)
    print(f'dim={dim} hidden={hid} heads={H} kv_heads={KV} vocab={V} seq={S} shared={shared}: {S} positions bitwise identical={ok} max|diff|={md}')
