import subprocess, sys, json, time
fx = json.load(open('fixtures.json'))
p = subprocess.Popen([sys.executable, 'run.py', 'model.bin', 'tokenizer.bin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
rd = lambda: p.stdout.readline().rstrip('\n')
def ask(s): p.stdin.write(s + '\n'); p.stdin.flush(); return rd()
def gen(s):
    p.stdin.write(s + '\n'); p.stdin.flush(); ids = []
    while True:
        l = rd()
        if l.startswith('T '): ids.append(int(l[2:]))
        else: return ids, l
assert rd() == 'READY'
# PIECE vs fixtures
bad = 0
for pr in fx['prompts']:
    seq = pr['seq']
    for i, want in enumerate(pr['pieces']):
        got = ask(f'PIECE {seq[i]} {seq[i+1]}')
        bad += got != want
print('PIECE mismatches:', bad)
print('PIECE byte tok:', ask('PIECE 5 3'), ask('PIECE 5 13'), ask('PIECE 1 29871'))
print('ENCODE empty-no-space:', ask('ENCODE'))
print('GENERATE 0:', gen('GENERATE 0 Once upon a time'))
print('GENERATE empty prompt:', gen('GENERATE 5 ')[1], gen('GENERATE 5')[1])
# FIXED argmax consistency with GENERATE
pr = fx['prompts'][0]
r = ask('FIXED ' + ' '.join(map(str, pr['seq']))).split()
n0 = len(pr['seq']) - len(pr['ids'])
print('FIXED argmax consistent:', r[1] == str(len(pr['seq'])) and [int(v) for v in r[2+n0-1:2+len(pr['seq'])-1]] == pr['ids'])
# context: long prompt of ~1100 tokens
text = ' '.join(['the cat sat on the mat and'] * 160)
ids = ask('ENCODE ' + text).split()[1:]
print('long prompt tokens:', len(ids))
t = time.time(); out = gen('GENERATE 50 ' + text); print('long prompt:', out, '%.1fs' % (time.time() - t))
short = ' '.join(['the cat sat on the mat and'] * 145)
n = len(ask('ENCODE ' + short).split()) - 1
t = time.time(); out = gen('GENERATE 100 ' + short); print(f'prompt {n} tokens ->', len(out[0]), out[1], 'expected new tokens if no stop:', 1024 - n + 1, '%.1fs' % (time.time() - t))
ask('QUIT') if False else (p.stdin.write('QUIT\n'), p.stdin.flush())
print('exit', p.wait())
