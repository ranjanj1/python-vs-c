"""Partial KV reset: after any command the cache must be all-zero again, and results must not depend on history."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, run
fx = json.load(open('fixtures.json'))
m = run.Model('model.bin')
def seqrun(seq):
    m.reset(); out = np.empty((len(seq), m.vocab), np.float32)
    for p, t in enumerate(seq): out[p] = m.forward(t, p)
    return out
short = fx['prompts'][0]['seq'][:20]
fresh = seqrun(short)
long_ = fx['prompts'][2]['seq'] + fx['prompts'][1]['seq']          # 161 positions
seqrun(long_)
again = seqrun(short)                                              # after a longer command
m.reset()
print('cache all-zero after reset:', not m.key_cache.any() and not m.value_cache.any(), ' used =', m.used)
print('short sequence identical after a longer one:', np.array_equal(fresh, again))
