# Compare the heap-based merge against the original O(n^2) run.c-style loop (prof/run_v1.py).
import sys, random, time
sys.path.insert(0, '.')
import run
sys.path.insert(0, 'prof')
import run_v1
new = run.Tokenizer('tokenizer.bin', 32000); old = run_v1.Tokenizer('tokenizer.bin', 32000)
import json
cases = list(json.load(open('fixtures.json'))['tokenizer'].keys())
rnd = random.Random(1)
words = [v.decode('utf-8','ignore').strip() for v in old.vocab[300:] if v.strip()]
alphabet = 'abcdefghijklmnopqrstuvwxyz  ,.!?\t\'"0123456789éüñ日本語😀\n'
for _ in range(3000):
    k = rnd.randrange(4)
    if k == 0: s = ''.join(rnd.choice(alphabet) for _ in range(rnd.randrange(60)))
    elif k == 1: s = ' '.join(rnd.choice(words) for _ in range(rnd.randrange(25)))
    elif k == 2: s = (rnd.choice(words) + rnd.choice(['', ' ', '  '])) * rnd.randrange(1, 30)
    else: s = None; b = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 40))).replace(b'\n', b'').replace(b'\0', b'')
    cases.append(s if s is not None else b)
bad = 0
for c in cases:
    b = c if isinstance(c, bytes) else c.encode()
    if new.encode(b) != old.encode(b): bad += 1; print('MISMATCH', repr(b)[:80])
print(len(cases), 'cases,', bad, 'mismatches')
for txt in [' '.join(['cat']*1020), 'Lily went to the park and played with her friends. '*60]:
    b = txt.encode(); t = time.perf_counter(); a = new.encode(b); t1 = time.perf_counter(); o = old.encode(b); t2 = time.perf_counter()
    print(len(a), 'ids: new %.1f ms, old %.1f ms, equal %s' % ((t1-t)*1e3, (t2-t1)*1e3, a == o))
