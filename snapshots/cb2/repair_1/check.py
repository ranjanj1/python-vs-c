#!/usr/bin/env python3
"""Public correctness check for your program. Standard library only. Passing this is necessary, not sufficient: the final gate uses other inputs.

  python3 check.py [--tokens N] [--prompts K] -- <your command>
Default is a quick check: first prompt, first 8 new tokens. Your command is given model.bin tokenizer.bin as arguments.
"""
import array, json, math, os, subprocess, sys, tempfile

argv = sys.argv[1:]
cut = argv.index('--')
opts, cmd = argv[:cut], argv[cut + 1:]
n_new = int(opts[opts.index('--tokens') + 1]) if '--tokens' in opts else 8
n_pr = int(opts[opts.index('--prompts') + 1]) if '--prompts' in opts else 1
fx = json.load(open('fixtures.json'))
V, tol = fx['vocab'], fx['tolerance']
ref = array.array('f'); ref.frombytes(open('fixtures.f32', 'rb').read())
p = subprocess.Popen(cmd + ['model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rd():
    s = p.stdout.readline()
    if not s:
        sys.exit(f'FAIL: program closed, exit {p.poll()}')
    return s.rstrip('\n')
def ask(s):
    p.stdin.write(s + '\n'); p.stdin.flush(); return rd()
if rd() != 'READY':
    sys.exit('FAIL: expected READY')
bad = 0
for text, want in fx['tokenizer'].items():
    if ask(f'ENCODE {text}') != want:
        print('FAIL tokenizer', repr(text)); bad += 1
for pr in fx['prompts'][:n_pr]:
    p.stdin.write(f"GENERATE {n_new} {pr['text']}\n"); p.stdin.flush()
    got = []
    while True:
        s = rd()
        if s.startswith('T '):
            got.append(int(s[2:]))
        else:
            break
    if got != pr['ids'][:n_new] or s.split()[1] != str(len(got)):
        print('FAIL generate', got[:10], s); bad += 1
    last = pr['positions'][-1] if n_new >= 8 else pr['positions'][1]
    seq = pr['seq'][:last + 1]
    with tempfile.TemporaryDirectory() as t:
        f = f'{t}/l.f32'
        if ask(f"LOGITS {f} " + ' '.join(map(str, seq))).split() != ['DONE', str(len(seq))]:
            print('FAIL LOGITS answer'); bad += 1; continue
        got = array.array('f'); got.frombytes(open(f, 'rb').read())
    if len(got) != len(seq) * V:
        print('FAIL logits size'); bad += 1; continue
    for k, pos in enumerate(pr['positions']):
        if pos > last:
            continue
        a = got[pos * V:(pos + 1) * V]; b = ref[(pr['row'] + k) * V:(pr['row'] + k + 1) * V]
        e = max((abs(x - y) if math.isfinite(x) else math.inf) for x, y in zip(a, b))
        print(f'position {pos}: max logit error {e:.3g}')
        if e > tol:
            print('FAIL logits'); bad += 1
ask_q = p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
print('PASS' if not bad and p.returncode == 0 else f'FAIL ({bad})')
sys.exit(1 if bad or p.returncode else 0)
