"""Randomised complete blocks, one implementation at a time. Appends every sample to results/raw/<run>.jsonl.

  python bench/measure.py <run-name> <blocks> <seed> [--official] ID=<cmd...> ID=<cmd...>   (quote each ID=cmd)
Per block and implementation: first use (fresh process), frozen warm-ups, warm request, fixed-token throughput, peak memory.
--official refuses to start off mains power. Pilots stay marked as pilots and never enter the final dataset.
"""
import hashlib, json, os, random, shlex, subprocess, sys, threading, time
sys.path.insert(0, 'bench')
from client import Proc
from workloads import DEV, EVAL, MODELS, TOKENIZER, NEW_TOKENS, EXPERIMENT_MODEL

LIMIT_S = 1800    # per request
WARMUPS = {'default': 1}   # the first-use request plus one more; PyPy pilot was stable from its first request
SETTLE_S = 30     # pause between samples, raised from 5 at Ranjan's request so heat from one sample does not carry into the next
SECS = lambda ns: ns / 1e9


def state():
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return {'power': sh("pmset -g batt | head -1 | cut -d\\' -f2"), 'therm': sh('pmset -g therm | tail -3 | tr -s " \\n" " "'),
            'free_pct': sh('memory_pressure | tail -1 | grep -o "[0-9]*%"'), 'swap': sh('sysctl -n vm.swapusage | cut -c1-60')}


def sample(impl, cmd, prompt, seq, warmups):
    row = {'impl': impl}
    p = Proc(cmd, MODELS[EXPERIMENT_MODEL], TOKENIZER)
    guard = threading.Timer(LIMIT_S * (3 + warmups), p.p.kill); guard.start()
    try:
        row['startup_s'] = SECS(p.t_ready - p.t_launch)
        ids, why, st, t0, t1 = p.generate(NEW_TOKENS, prompt)
        row['first_use_ttft_s'] = SECS(st[0] - p.t_launch)
        for _ in range(warmups):
            p.generate(NEW_TOKENS, prompt)
        ids2, why2, st, t0, t1 = p.generate(NEW_TOKENS, prompt)
        row.update(ready_ttft_s=SECS(st[0] - t0), request_s=SECS(t1 - t0), new_tokens=len(ids2), stop=why2, same_as_first=ids == ids2)
        row['decode_tok_s'] = (len(st) - 1) / SECS(st[-1] - st[0]) if len(st) > 1 else None
        am, ns = p.fixed(seq)
        row.update(fixed_tok_s=len(seq) / SECS(ns), fixed_argmax_hash=hashlib.sha256(str(am).encode()).hexdigest()[:12])
    except Exception as e:
        row['failed'] = f'{type(e).__name__}: {str(e)[:120]}'
    finally:
        guard.cancel()
    try:
        row['exit'], row['peak_rss_bytes'] = p.close()
        row.update(cpu_s=p.cpu_s, wall_s=p.wall_s, cpu_per_wall=p.cpu_s / p.wall_s)
    except Exception as e:
        row['failed'] = row.get('failed', f'close: {e}')
    return row


def main():
    a = sys.argv[1:]
    official = '--official' in a
    a = [x for x in a if x != '--official']
    run, blocks, seed = a[0], int(a[1]), int(a[2])
    impls = {x.split('=', 1)[0]: shlex.split(x.split('=', 1)[1]) for x in a[3:]}
    if official and 'AC Power' not in state()['power']:
        sys.exit('official run refused: not on mains power')
    rng = random.Random(seed)
    prompt = (EVAL if official else DEV)[0]
    ref = Proc(['ports/c0/run'], MODELS[EXPERIMENT_MODEL], TOKENIZER)
    ids, *_ = ref.generate(NEW_TOKENS, prompt)
    seq = list(map(int, ref.ask(f'ENCODE {prompt}').split()[1:])) + ids
    ref.close()
    os.makedirs('results/raw', exist_ok=True)
    with open(f'results/raw/{run}.jsonl', 'a', buffering=1) as f:
        for b in range(blocks):
            order = list(impls); rng.shuffle(order)
            for k, impl in enumerate(order):
                time.sleep(SETTLE_S)
                if official and 'No thermal warning' not in state()['therm']:
                    sys.exit('official run stopped: thermal warning')
                row = sample(impl, impls[impl], prompt, seq, WARMUPS.get(impl, WARMUPS['default']))
                row.update(run=run, official=official, block=b, slot=k, seed=seed, time=time.time(), state=state())
                f.write(json.dumps(row) + '\n')
                print(b, impl, {x: round(row[x], 3) for x in ('decode_tok_s', 'fixed_tok_s') if isinstance(row.get(x), float)}, row.get('failed', ''), flush=True)

main()
