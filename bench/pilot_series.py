"""Pilot only. Warm-up series of FIXED requests in one process, and a sustained-load drift check on C0. Output is never a result."""
import json, sys, time
sys.path.insert(0, 'bench')
from client import Proc
from workloads import MODELS, TOKENIZER
M = MODELS['110M']; out = open('results/raw/pilot-series.jsonl', 'a', buffering=1)
PP = sys.argv[1]
seq = [1, 9038, 2501, 263, 931, 29892, 727, 471]
def series(name, cmd, reps):
    p = Proc(cmd, M, TOKENIZER, timeout=1800)
    ready = (p.t_ready - p.t_launch) / 1e9
    rates = []
    for i in range(reps):
        _, ns = p.fixed(seq); rates.append(round(len(seq) / (ns / 1e9), 4))
        print(name, i, rates[-1], flush=True)
    code, rss = p.close()
    out.write(json.dumps({'pilot': name, 'ready_s': ready, 'tok_s': rates, 'rss': rss, 'exit': code}) + '\n')
series('pure-cpython', ['/Users/Shared/pvc-kit/stdlib/bin/python', '-I', 'controls/pure_python_control.py'], 3)
series('pure-pypy', [PP, 'controls/pure_python_control.py'], 12)
p = Proc(['ports/c0/run'], M, TOKENIZER); long = seq * 16; r = []
end = time.time() + 480
while time.time() < end:
    _, ns = p.fixed(long); r.append(round(len(long) / (ns / 1e9), 3))
p.close()
out.write(json.dumps({'pilot': 'c0-sustained-8min', 'tok_s': r}) + '\n')
print('sustained first/last', r[:3], r[-3:], 'min', min(r), 'max', max(r), 'n', len(r), flush=True)
