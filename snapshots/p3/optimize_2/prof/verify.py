# Compare run.py against the plain round-1 implementation (prof/run_v1.py) on
#  (1) the real model over a full 1024-position sequence (exercises both weighted-sum branches);
#  (2) synthetic random checkpoints: GQA, unshared classifier, block-indivisible dims.
import sys, struct, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'prof')
import run, run_v1
def compare(path, ids, tag):
    a = run.Transformer(path); b = run_v1.Transformer(path)
    worst = 0.0; arg_mismatch = 0
    for p, t in enumerate(ids):
        la = a.forward(t, p).copy(); lb = b.forward(t, p)
        worst = max(worst, float(np.abs(la - lb).max())); arg_mismatch += int(np.argmax(la) != np.argmax(lb))
    # second command must start clean: rerun a prefix after reset
    a.reset(); b.reset()
    for p, t in enumerate(ids[:20]):
        la = a.forward(t, p).copy(); lb = b.forward(t, p); worst = max(worst, float(np.abs(la - lb).max()))
    print('%-34s positions %4d  max |dlogit| %.2e  argmax mismatches %d' % (tag, len(ids), worst, arg_mismatch))
rng = np.random.default_rng(0)
ids = [1] + list(rng.integers(3, 32000, 1023))
compare('model.bin', ids, 'real model, 1024 random ids')
def synth(name, dim, hid, L, H, KVH, V, S, shared):
    hs = dim // H; kv = dim * KVH // H
    parts = [rng.standard_normal(V * dim) * 0.5, 1 + 0.1 * rng.standard_normal(L * dim),
             rng.standard_normal(L * dim * dim) / np.sqrt(dim), rng.standard_normal(L * dim * kv) / np.sqrt(dim),
             rng.standard_normal(L * dim * kv) / np.sqrt(dim), rng.standard_normal(L * dim * dim) / np.sqrt(dim),
             1 + 0.1 * rng.standard_normal(L * dim), rng.standard_normal(L * dim * hid) / np.sqrt(dim),
             rng.standard_normal(L * hid * dim) / np.sqrt(hid), rng.standard_normal(L * dim * hid) / np.sqrt(dim),
             1 + 0.1 * rng.standard_normal(dim), np.zeros(S * hs)]
    if not shared: parts.append(rng.standard_normal(V * dim) * 0.5)
    path = 'prof/synth/%s.bin' % name
    with open(path, 'wb') as f:
        f.write(struct.pack('<7i', dim, hid, L, H, KVH, V if shared else -V, S))
        f.write(np.concatenate(parts).astype('<f4').tobytes())
    return path, V, S
for args in [('gqa_unshared', 128, 384, 2, 8, 2, 512, 900, False),
             ('odd_dims', 288, 768, 3, 6, 6, 1000, 820, True),
             ('mqa_odd', 120, 200, 2, 4, 1, 300, 800, False)]:
    path, V, S = synth(*args)
    compare(path, [1] + list(rng.integers(3, V, S - 1)), '%s (dim %d hid %d H %d KV %d)' % (args[0], args[1], args[2], args[4], args[5]))
