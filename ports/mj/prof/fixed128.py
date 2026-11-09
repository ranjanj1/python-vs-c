import subprocess, sys, time
exe, d = sys.argv[1], sys.argv[2]
p = subprocess.Popen([exe, d + '/model.bin', d + '/tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
assert p.stdout.readline().strip() == 'READY'
ts = []
for r in range(5):
    t = time.time(); p.stdin.write('FIXED ' + ' '.join(['1'] + ['300'] * 127) + '\n'); p.stdin.flush(); p.stdout.readline(); ts.append((time.time() - t) / 128 * 1e3)
print(f'ms/position: best {min(ts):.3f} mean {sum(ts)/len(ts):.3f}')
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
