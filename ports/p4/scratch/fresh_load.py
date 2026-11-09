import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
mode = sys.argv[1]
if mode == 'memmap':
    np.fromfile = lambda path, dtype, offset: np.memmap(path, dtype=dtype, mode='r', offset=offset)
import run
t = time.perf_counter(); m = run.Model('model.bin'); print(f'{mode:8s} Model() {time.perf_counter()-t:.3f}s')
