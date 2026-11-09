# Interleaved A/B: all variants loaded at once, alternate short FIXED-64 rounds, report min/median ms per position.
import subprocess, sys, time, statistics
d = sys.argv[1]; exes = sys.argv[2:]; rounds = 12
ps = []
for e in exes:
    p = subprocess.Popen([e, d + '/model.bin', d + '/tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert p.stdout.readline().strip() == 'READY'; ps.append(p)
cmd = 'FIXED ' + ' '.join(['1'] + ['300'] * 63) + '\n'
ts = [[] for _ in exes]
for r in range(rounds):
    for i, p in enumerate(ps if r % 2 == 0 else ps[::-1]):
        k = i if r % 2 == 0 else len(ps) - 1 - i
        t = time.perf_counter(); p.stdin.write(cmd); p.stdin.flush(); p.stdout.readline(); ts[k].append((time.perf_counter() - t) / 64 * 1e3)
for e, t, p in zip(exes, ts, ps):
    print(f'{e.split("/")[-1]:>12}: min {min(t):.3f}  median {statistics.median(t):.3f} ms/position')
    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
