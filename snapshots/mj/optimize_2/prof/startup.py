import subprocess, sys, time
exe, d = sys.argv[1], sys.argv[2]
ts = []
for _ in range(5):
    t = time.time()
    p = subprocess.Popen([exe, d + '/model.bin', d + '/tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert p.stdout.readline().strip() == 'READY'; ts.append(time.time() - t)
    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
print(exe, 'time to READY: min %.3f median %.3f s' % (min(ts), sorted(ts)[2]))
