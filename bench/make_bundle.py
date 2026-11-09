"""Builds the folder an agent session receives: reference source, rules, 110M model, tokenizer, stdlib check.py and public fixtures.

  python bench/make_bundle.py <out-folder>     (run from the experiment folder)
EVAL prompts are never included.
"""
import json, os, shutil, sys
sys.path.insert(0, 'bench')
import numpy as np
from client import Proc
from workloads import DEV, MODELS, TOKENIZER, TOKENIZER_CASES, NEW_TOKENS, TOL_MAX_ABS, EXPERIMENT_MODEL

out = sys.argv[1]
os.makedirs(f'{out}/reference', exist_ok=True)
for src, dst in [('sources/upstream/run.c', 'reference/run.c'), ('sources/upstream/LICENSE', 'reference/LICENSE'), ('RULES.md', 'RULES.md'),
                 ('CLAUDE.md', 'CLAUDE.md'), ('bench/INTERFACE.md', 'INTERFACE.md'), ('bench/check.py', 'check.py'),
                 (MODELS[EXPERIMENT_MODEL], 'model.bin'), (TOKENIZER, 'tokenizer.bin')]:
    shutil.copy(src, f'{out}/{dst}')
c = Proc(['ports/c0/run'], MODELS[EXPERIMENT_MODEL], TOKENIZER)
fx = {'tolerance': TOL_MAX_ABS, 'vocab': 32000, 'tokenizer': {t: c.ask(f'ENCODE {t}') for t in TOKENIZER_CASES}, 'prompts': []}
logit_rows = []
for text in DEV:
    ids, why, *_ = c.generate(NEW_TOKENS, text)
    enc = list(map(int, c.ask(f'ENCODE {text}').split()[1:]))
    seq = enc + ids
    path = f'{out}/.tmp.f32'
    c.ask(f'LOGITS {path} ' + ' '.join(map(str, seq)))
    a = np.fromfile(path, '<f4').reshape(len(seq), 32000); os.remove(path)
    pos = [0, len(enc) - 1, len(enc) + 7]
    fx['prompts'].append({'text': text, 'ids': ids, 'stop': why, 'seq': seq, 'positions': pos, 'row': len(logit_rows),
                          'pieces': [c.ask(f'PIECE {x} {y}') for x, y in zip(seq, seq[1:])]})
    logit_rows += [a[i] for i in pos]
c.close()
json.dump(fx, open(f'{out}/fixtures.json', 'w'))
np.stack(logit_rows).astype('<f4').tofile(f'{out}/fixtures.f32')
print(out, sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))) >> 20, 'MB')
