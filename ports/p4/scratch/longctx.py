"""Long-context timing: GENERATE after a ~1016-token prompt, and FIXED over 300 / 1024 positions (synthetic token ids
taken from the dev prompt texts only). Usage: python3 scratch/longctx.py <program.py>"""
import os, sys, subprocess, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
fx = json.load(open('fixtures.json'))
p = subprocess.Popen([sys.executable, sys.argv[1], 'model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
rd = lambda: p.stdout.readline().rstrip('\n')
assert rd() == 'READY'
base = []
for pr in fx['prompts']: base += pr['seq']
ids = (base * 20)[:1024]
out = {}
for n in (300, 1024):
    t = time.perf_counter(); p.stdin.write('FIXED ' + ' '.join(map(str, ids[:n])) + '\n'); p.stdin.flush(); r = rd()
    out[f'FIXED {n}'] = time.perf_counter() - t
    import hashlib; out[f'FIXED {n} answer sha1'] = hashlib.sha1(r.encode()).hexdigest()[:12]
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
print(sys.argv[1], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
