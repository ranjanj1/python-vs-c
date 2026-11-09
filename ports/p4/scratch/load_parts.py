"""Fresh process: split weight-prep cost into file-page faults, anonymous-page faults, and copy work."""
import time, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from numba import njit
import run
@njit(cache=True)
def touch_read(a, step):
    s = np.float32(0)
    for i in range(0, a.shape[0], step): s += a[i]
    return s
@njit(cache=True)
def touch_write(a, step):
    for i in range(0, a.shape[0], step): a[i] = np.float32(0)
touch_read(np.zeros(8, np.float32), 1); touch_write(np.zeros(8, np.float32), 1)
src = np.zeros((1, 128, 16), np.float32); dst = np.empty((1, 1, 16, 128), np.float32); run.tile_into(dst, 0, src)
mm = np.memmap('model.bin', dtype='<f4', mode='r', offset=28)
t = time.perf_counter(); touch_read(mm, 4096); a = time.perf_counter() - t
t = time.perf_counter(); touch_read(mm, 4096); a2 = time.perf_counter() - t
anon = np.empty(536_000_000 // 4, np.float32)
t = time.perf_counter(); touch_write(anon, 4096); b = time.perf_counter() - t
t = time.perf_counter(); touch_write(anon, 4096); b2 = time.perf_counter() - t
del anon
print(f'file pages first touch {a*1e3:6.1f} ms (again {a2*1e3:.1f})   anon 536MB first touch {b*1e3:6.1f} ms (again {b2*1e3:.1f})')
t = time.perf_counter(); m = run.Model('model.bin'); print(f'Model() after file pages resident: {1e3*(time.perf_counter()-t):.1f} ms')
