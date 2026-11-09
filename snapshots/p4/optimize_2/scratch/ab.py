"""A/B: scratch/run_v1.py (before) vs run.py (after) on the dev sequences: bitwise logits + interleaved timing."""
import os, sys, time, statistics, json, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
OLD = sys.argv[2] if len(sys.argv) > 2 else 'run_v1'
old = load(OLD, os.path.join(ROOT, 'scratch', OLD + '.py'))
new = load('run', os.path.join(ROOT, 'run.py'))
fx = json.load(open('fixtures.json'))
mo = old.Model('model.bin'); mn = new.Model('model.bin')
def seqrun(m, seq):
    m.reset(); out = np.empty((len(seq), m.vocab), np.float32)
    for p, t in enumerate(seq): out[p] = m.forward(t, p)
    return out
for k, pr in enumerate(fx['prompts']):
    a = seqrun(mo, pr['seq']); b = seqrun(mn, pr['seq'])
    print(f'prompt {k}: {len(pr["seq"])} positions, logits bitwise identical: {np.array_equal(a, b)}  max|diff| {float(np.abs(a-b).max()):.3g}')
seq = fx['prompts'][2]['seq']
R = {'before': [], 'after': []}
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
    for name, m in (('before', mo), ('after', mn)):
        t = time.perf_counter(); seqrun(m, seq); R[name].append((time.perf_counter() - t) / len(seq))
for k, v in R.items():
    print(f'{k:7s} median {statistics.median(v)*1e3:.3f} ms/token  min {min(v)*1e3:.3f}')
print(f'speedup (median) {statistics.median(R["before"]) / statistics.median(R["after"]):.3f}x')
