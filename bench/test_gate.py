"""The gate must pass known-correct code and reject every cheat. Run from the experiment folder. Exit 1 on any surprise."""
import os, shutil, sys, tempfile
sys.path.insert(0, 'bench')
from validate import run
from workloads import DEV, MODELS, TOKENIZER

PY = sys.executable
FAKES = ['nan', 'zero_compute', 'noise', 'truncated_logits', 'skip_last_position', 'lazy_fixed', 'short_fixed', 'hardcoded', 'empty',
         'wrong_count', 'truncated_gen', 'wrong_tokens', 'bad_exit', 'dies_at_start']
ENV = {'PYTHONPATH': ''}
fail = 0
def expect(label, res, want):
    global fail
    ok = res['eligible'] == want
    fail += not ok
    print(('ok  ' if ok else 'BAD ') + f"{label}: {res['numeric']}, generation {res['exact_generation']}, {[e for r in res['models'] for e in r['errors']][:1]}", flush=True)

expect('reference vs itself', run('c0', ['ports/c0/run'], DEV, ['15M', '110M']), True)
expect('numpy control float32', run('np32', [PY, '-I', 'controls/numpy_control.py'], DEV, ['15M'], numeric_only=True, env={'VECLIB_MAXIMUM_THREADS': '1'}), True)
for m in FAKES:
    expect(f'fake {m}', run(m, [PY, '-I', 'bench/fakes/fake.py', m], DEV, ['15M']), False)
with tempfile.TemporaryDirectory() as t:
    b = bytearray(open(MODELS['15M'], 'rb').read()); b[5_000_000:5_000_400] = bytes(400)
    open(f'{t}/bad.bin', 'wb').write(b)
    expect('corrupted weights', run('corrupt', ['ports/c0/run'], DEV, ['15M'], model_path=f'{t}/bad.bin'), False)
    b = bytearray(open(TOKENIZER, 'rb').read()); b[4000:4040] = b'x' * 40
    open(f'{t}/broken.tok', 'wb').write(b)
    expect('unreadable tokenizer', run('badtok', ['ports/c0/run'], DEV, ['15M'], tok_path=f'{t}/broken.tok'), False)
    b = bytearray(open(TOKENIZER, 'rb').read()); pos = 4
    for tok in range(931):   # walk to token 931, the exact word "time": float score, int length, bytes
        pos += 8 + int.from_bytes(b[pos + 4:pos + 8], 'little')
    n = int.from_bytes(b[pos + 4:pos + 8], 'little'); assert bytes(b[pos + 8:pos + 8 + n]) == b' time', b[pos + 8:pos + 8 + n]
    b[pos + 9:pos + 13] = b'tide'   # loads fine, one word is wrong
    open(f'{t}/swapped.tok', 'wb').write(b)
    expect('swapped tokenizer word', run('swaptok', ['ports/c0/run'], DEV, ['15M'], tok_path=f'{t}/swapped.tok'), False)
print('GATE', 'FAILED' if fail else 'PASSED', fail)
sys.exit(1 if fail else 0)
