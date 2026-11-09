# In-situ forward time at position 80 / 600 for different column-block sizes (monkeypatched).
import sys, time
sys.path.insert(0, '.')
import run
b = int(sys.argv[1]); run.block_size = lambda n: b if n % b == 0 else n
m = run.Transformer('model.bin')
for p in range(600): m.forward(5 + p % 300, p)
res = []
for pos in (80, 599):
    best = 1e9
    for r in range(5):
        t = time.perf_counter()
        for _ in range(15): m.forward(7, pos)
        best = min(best, (time.perf_counter() - t) / 15)
    res.append('pos %d %.2f ms' % (pos, best * 1e3))
print('block %d: ' % b + ' | '.join(res))
