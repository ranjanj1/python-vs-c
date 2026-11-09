"""Fresh-process startup breakdown of run.py up to READY (same steps as run.main)."""
import time; T0 = time.perf_counter()
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
marks = []
def mark(name): marks.append((name, time.perf_counter()))
import numpy; mark('import numpy')
import numba; mark('import numba')
import run; mark('import run')
m = run.Model.__new__(run.Model)
# replicate Model.__init__ with timing around the first tile_into call (includes cache load)
orig = run.tile_into; first = [True]
def ti(*a):
    if first[0]:
        first[0] = False; t = time.perf_counter(); orig(*a); marks.append(('  first tile_into call (cache load+run)', time.perf_counter())); return
    return orig(*a)
run.tile_into = ti
run.Model.__init__(m, 'model.bin'); mark('Model() load+tile')
tok = run.Tokenizer('tokenizer.bin', m.vocab); mark('Tokenizer()')
m.forward(1, 0); mark('first forward (cache load+run)')
m.forward(1, 1); mark('second forward')
run.argmax(m.logits); mark('argmax first call')
m.reset(); mark('reset (zero KV cache)')
prev = T0
for name, t in marks:
    print(f'{name:42s} {1e3*(t-prev):8.1f} ms   (cum {1e3*(t-T0):7.1f})'); prev = t
