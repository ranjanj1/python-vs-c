# Wall time from process start to READY (min/median of N fresh starts).
import subprocess, sys, time, statistics
prog = sys.argv[1:]; ts = []
for _ in range(7):
    t = time.perf_counter()
    p = subprocess.Popen(prog + ['model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'READY'; ts.append(time.perf_counter() - t)
    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
print('%s: READY min %.3fs median %.3fs' % (prog[-1], min(ts), statistics.median(ts)))
