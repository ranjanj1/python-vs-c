"""Correctness gate. Compares one implementation with the pinned C reference. Speed is never measured here.

  python bench/validate.py --name X [--set dev|eval] [--models 15M,110M] [--numeric-only] -- <command...>

Statuses are separate: numeric (logits within tolerance), tokenizer, exact_generation (matched/diverged), eligible.
Run from the experiment folder.
"""
import argparse, json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from client import Proc
from workloads import DEV, EVAL, MODELS, TOKENIZER, TOKENIZER_CASES, NEW_TOKENS, TOL_MAX_ABS, LONG_POSITIONS

REF = ['ports/c0/run']


def ids_of(p, text):
    a = p.ask(f'ENCODE {text}').split()
    if not a or a[0] != 'IDS':
        raise RuntimeError(f'bad ENCODE answer {a[:2]}')
    return list(map(int, a[1:]))


def logits(p, seq, path, vocab):
    if os.path.exists(path):
        os.remove(path)
    a = p.ask(f'LOGITS {path} ' + ' '.join(map(str, seq)))
    if a.split() != ['DONE', str(len(seq))]:
        raise RuntimeError(f'bad LOGITS answer {a[:40]}')
    if os.path.getsize(path) != len(seq) * vocab * 4:
        raise RuntimeError(f'logits file size {os.path.getsize(path)}')
    return np.fromfile(path, '<f4').reshape(len(seq), vocab)


def check_model(cmd, model_key, prompts, numeric_only, tmp, env, model_path=None, tok_path=None):
    r = {'model': model_key, 'errors': [], 'prompts': []}
    mp, tp = MODELS[model_key], TOKENIZER
    ref, vocab = Proc(REF, mp, tp), 32000
    try:
        c = Proc(cmd, model_path or mp, tok_path or tp, env=env, timeout=1800)
    except Exception as e:
        ref.close()
        r['errors'].append(f'failed to start: {str(e)[:160]}')
        return r
    try:
        if not numeric_only:
            bad = [t for t in TOKENIZER_CASES if ids_of(ref, t) != ids_of(c, t)]
            if bad:
                r['errors'].append(f'tokenizer differs on {bad[:3]}')
            wrong = [b for a in (0, 1) for b in range(vocab) if ref.ask(f'PIECE {a} {b}') != c.ask(f'PIECE {a} {b}')]   # whole vocabulary
            if wrong:
                r['errors'].append(f'decode differs for {len(wrong)} tokens, first {wrong[0]}')
        worst = 0.0
        for text in prompts:
            ids, why, *_ = ref.generate(NEW_TOKENS, text)
            seq = (ids_of(ref, text) + ids)[:256]
            a, b = logits(ref, seq, f'{tmp}/ref.f32', vocab), logits(c, seq, f'{tmp}/cand.f32', vocab)
            if not np.isfinite(b).all():
                r['errors'].append('non-finite logits'); continue
            err = float(np.abs(a - b).max()); worst = max(worst, err)
            if err > TOL_MAX_ABS:
                r['errors'].append(f'logits off by {err:.3g}')
            top = np.sort(a, 1); margin = top[:, -1] - top[:, -2]
            am_ref, am_c = a.argmax(1), b.argmax(1)
            am_fixed, _ = c.fixed(seq)
            if am_fixed != am_c.tolist():
                r['errors'].append('FIXED does not agree with LOGITS')
            real = (am_ref != am_c) & (margin > 2 * TOL_MAX_ABS)
            if real.any():
                r['errors'].append(f'argmax differs at {int(real.sum())} clear positions')
            row = {'prompt_tokens': len(ids_of(ref, text)) if not numeric_only else None, 'new': len(ids), 'stop': why, 'max_err': err}
            if not numeric_only:
                got, gwhy, *_ = c.generate(NEW_TOKENS, text)
                row['exact'] = got == ids and gwhy == why
                if not row['exact']:
                    i = next((k for k, (x, y) in enumerate(zip(ids, got)) if x != y), min(len(ids), len(got)))
                    row['first_divergence'] = i
                    pos = len(ids_of(ref, text)) - 1 + i
                    row['margin_there'] = float(margin[pos]) if pos < len(margin) else None
                    if not got:
                        r['errors'].append('empty generation')
                    elif row['margin_there'] is None or row['margin_there'] > 2 * TOL_MAX_ABS:
                        r['errors'].append(f'generation diverged at token {i} with a clear margin')
            r['prompts'].append(row)
        base = [t for text in prompts for t in ids_of(ref, text)] + ids
        long = [1] + (base * (LONG_POSITIONS // len(base) + 1))[:LONG_POSITIONS - 1]
        a, b = logits(ref, long, f'{tmp}/ref.f32', vocab), logits(c, long, f'{tmp}/cand.f32', vocab)
        err = float(np.abs(a - b).max()) if np.isfinite(b).all() else math.inf
        worst = max(worst, err)
        if err > TOL_MAX_ABS:
            r['errors'].append(f'long-sequence logits off by {err:.3g}')
        r['max_err'] = worst
    except Exception as e:
        r['errors'].append(f'{type(e).__name__}: {str(e)[:160]}')
    finally:
        ref.close()
        try:
            code, rss = c.close()
            if code != 0:
                r['errors'].append(f'exit status {code}')
        except Exception as e:
            r['errors'].append(f'close failed {e}')
    return r


def run(name, cmd, prompts, models, numeric_only=False, env=None, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        rows = [check_model(cmd, m, prompts, numeric_only, tmp, env, **kw) for m in models]
    numeric = all(not r['errors'] for r in rows)
    exact = None if numeric_only else all(p.get('exact') for r in rows for p in r['prompts']) and all(r['prompts'] for r in rows)
    return {'name': name, 'numeric': 'passed' if numeric else 'failed', 'exact_generation': {None: 'not checked', True: 'matched', False: 'diverged'}[exact],
            'eligible': numeric, 'models': rows}


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--name', required=True); a.add_argument('--set', default='dev'); a.add_argument('--models', default='110M')
    a.add_argument('--numeric-only', action='store_true'); a.add_argument('cmd', nargs='+')
    o = a.parse_args()
    out = run(o.name, o.cmd, DEV if o.set == 'dev' else EVAL, o.models.split(','), o.numeric_only)
    os.makedirs('validation', exist_ok=True)
    json.dump(out, open(f'validation/{o.name}-{o.set}.json', 'w'), indent=1)
    print(o.name, out['numeric'], out['exact_generation'], [r['errors'][:2] for r in out['models']])
