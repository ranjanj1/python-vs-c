# End-to-end protocol timing on the development prompts (fixtures.json): startup, GENERATE 64, LOGITS, long prompt.
import subprocess, sys, time, json, tempfile, os
prog = sys.argv[1:]
fx = json.load(open('fixtures.json'))
t0 = time.perf_counter()
p = subprocess.Popen(prog + ['model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
assert p.stdout.readline().strip() == 'READY'; t_ready = time.perf_counter() - t0
def run(cmd):
    p.stdin.write(cmd + '\n'); p.stdin.flush(); out = []
    while True:
        s = p.stdout.readline().strip(); out.append(s)
        if s.startswith('DONE'): return out
res = {'ready': t_ready}
t = time.perf_counter(); outs = [run('GENERATE 64 ' + pr['text']) for pr in fx['prompts']]; res['generate3x64'] = time.perf_counter() - t
assert all([int(s[2:]) for s in o[:-1]] == pr['ids'] for o, pr in zip(outs, fx['prompts']))
with tempfile.TemporaryDirectory() as d:
    t = time.perf_counter(); [run(f'LOGITS {d}/l.f32 ' + ' '.join(map(str, pr['seq']))) for pr in fx['prompts']]; res['logits3'] = time.perf_counter() - t
long = ' '.join(['cat'] * 1020)
t = time.perf_counter(); o = run('GENERATE 50 ' + long); res['long_prompt_1021'] = time.perf_counter() - t; assert o[-1] == 'DONE 4 context', o[-1]
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait(); res['total'] = time.perf_counter() - t0
print(' '.join('%s=%.2fs' % kv for kv in res.items()), 'exit', p.returncode)
