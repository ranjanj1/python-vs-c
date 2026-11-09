# Development workload: the fixture prompts, as check.py drives them (GENERATE 64 + LOGITS on seq), repeated.
import json, subprocess, sys, time, os, tempfile
exe, d, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1
fx = json.load(open(d + '/fixtures.json'))
p = subprocess.Popen([exe, d + '/model.bin', d + '/tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
rd = lambda: p.stdout.readline().rstrip('\n')
t0 = time.time(); assert rd() == 'READY'; tload = time.time() - t0
print(f'pid {p.pid} load {tload:.3f}s', flush=True)
tg = tl = 0; ng = nl = 0
tmp = tempfile.mkdtemp()
for r in range(reps):
    for pr in fx['prompts']:
        t = time.time(); p.stdin.write(f"GENERATE 64 {pr['text']}\n"); p.stdin.flush()
        got = []
        while True:
            s = rd()
            if s.startswith('T '): got.append(int(s[2:]))
            else: break
        tg += time.time() - t; ng += len(pr['seq'])
        assert got == pr['ids'][:64], 'generate mismatch'
        t = time.time(); r_ = (p.stdin.write(f"LOGITS {tmp}/l.f32 " + ' '.join(map(str, pr['seq'])) + '\n'), p.stdin.flush(), rd())[2]
        tl += time.time() - t; nl += len(pr['seq'])
print(f'GENERATE: {ng} positions {tg:.2f}s = {ng/tg:.1f} pos/s | LOGITS: {nl} positions {tl:.2f}s = {nl/tl:.1f} pos/s')
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
