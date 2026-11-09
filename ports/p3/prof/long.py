# Per-token forward time (min of repeats) at several positions, after filling the cache.
import sys, time, importlib
sys.path.insert(0, sys.argv[1]); mod = importlib.import_module(sys.argv[2])
m = mod.Transformer('model.bin')
for p in range(1000): m.forward(5 + p % 300, p)
out = []
for pos in (80, 300, 600, 999):
    best = 1e9
    for r in range(5):
        t = time.perf_counter()
        for _ in range(15): m.forward(7, pos)
        best = min(best, (time.perf_counter() - t) / 15)
    out.append('pos %d %.2f ms' % (pos, best * 1e3))
print(sys.argv[2] + ': ' + ' | '.join(out))
