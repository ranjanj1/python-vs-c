"""Cheating or broken programs for testing the gate. Wraps the real C reference and damages one thing. Argv: mode model tokenizer."""
import os, subprocess, sys
import numpy as np

mode, model, tok = sys.argv[1:4]
p = subprocess.Popen(['ports/c0/run', model, tok], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rd(): return p.stdout.readline().rstrip('\n')
def tx(s): p.stdin.write(s + '\n'); p.stdin.flush()
def out(s): print(s, flush=True)
CANNED = {}
out(rd())
if mode == 'dies_at_start':
    sys.exit(3)
for line in sys.stdin:
    line = line.rstrip('\n'); cmd = line.split(' ')[0]
    if cmd == 'QUIT':
        tx('QUIT'); p.wait()
        sys.exit(5 if mode == 'bad_exit' else 0)
    if cmd == 'ENCODE' and mode == 'wrong_tokens':
        tx(line); a = rd().split(); a[-1] = '13'; out(' '.join(a)); continue
    if cmd == 'LOGITS':
        tx(line); r = rd(); path = line.split(' ')[1]
        a = np.fromfile(path, '<f4')
        if mode == 'nan': a[100] = np.nan
        if mode == 'zero_compute': a[:] = 0
        if mode == 'noise': a += np.random.default_rng(1).normal(0, 1e-2, a.size).astype('<f4')
        if mode == 'truncated_logits': a = a[:-5]
        if mode == 'skip_last_position': a[-32000:] = a[-64000:-32000]
        a.tofile(path); out(r); continue
    if cmd == 'FIXED' and mode == 'lazy_fixed':
        n = len(line.split()) - 1; out('DONE %d %s' % (n, ' '.join(['1'] * n))); continue
    if cmd == 'FIXED' and mode == 'short_fixed':
        tx(line); a = rd().split(); out('DONE %d %s' % (int(a[1]) - 1, ' '.join(a[3:]))); continue
    if cmd == 'GENERATE':
        tx(line); n = 0; lines = []
        while True:
            r = rd(); lines.append(r)
            if r.startswith('DONE'): break
        if mode == 'hardcoded':   # the same canned story whatever the prompt: output matching alone must not pass
            lines = CANNED.setdefault('story', lines)
        if mode == 'empty': lines = ['DONE 0 stop']
        if mode == 'wrong_count': lines[-1] = 'DONE 7 max'
        if mode == 'truncated_gen': lines = lines[:20] + [f'DONE 20 {lines[-1].split()[2]}']
        for r in lines: out(r)
        continue
    tx(line); out(rd())
