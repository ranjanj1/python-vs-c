import os, sys, time, importlib.util, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
mods = [('v1', load('run_v1', 'scratch/run_v1.py')), ('new', load('run', 'run.py'))]
for name, mod in mods:  # compile/warm kernels on a tiny call
    m = mod.Model('model.bin'); del m; gc.collect()
for r in range(3):
    for name, mod in mods:
        t = time.perf_counter(); m = mod.Model('model.bin'); dt = time.perf_counter() - t
        del m; gc.collect()
        print(f'{name:4s} Model() {dt:.3f}s')
