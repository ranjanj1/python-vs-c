# One load strategy per fresh process: python3 prof/load1.py A|B|fromfile
import sys, time
t0 = time.perf_counter()
sys.argv = [sys.argv[0]] + sys.argv[1:]
import numpy as np
exec(open('prof/load.py').read().split("for name, fn in")[0])
t = time.perf_counter()
r = {'A': A, 'B': B, 'fromfile': lambda: np.fromfile(path, dtype=F, offset=28)}[sys.argv[1]]()
print('%-9s %.3fs' % (sys.argv[1], time.perf_counter() - t))
