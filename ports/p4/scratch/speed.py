import subprocess, sys, time
t0 = time.time()
p = subprocess.Popen([sys.executable, 'run.py', 'model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
assert p.stdout.readline().strip() == 'READY'
print('startup %.2fs' % (time.time() - t0))
for n in (64, 256, 1000):
    t = time.time(); p.stdin.write(f'GENERATE {n} Once upon a time\n'); p.stdin.flush()
    k = 0
    while True:
        s = p.stdout.readline()
        if s.startswith('T '): k += 1
        else: break
    dt = time.time() - t
    print(f'GENERATE {n}: {s.strip()} {dt:.2f}s  {k/dt:.1f} tok/s')
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
