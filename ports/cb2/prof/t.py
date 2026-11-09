import json, subprocess, time, sys, array
fx=json.load(open('fixtures.json'))
p=subprocess.Popen(['./cb2','model.bin','tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
t0=time.time(); assert p.stdout.readline().strip()=='READY'; print('load %.3fs'%(time.time()-t0))
def ask(s): p.stdin.write(s+'\n'); p.stdin.flush(); return p.stdout.readline().rstrip('\n')
bad=0
for pr in fx['prompts']:
    s=pr['seq']
    for k,want in enumerate(pr['pieces']):
        got=ask(f'PIECE {s[k]} {s[k+1]}')
        if got!=want: bad+=1; print('piece',s[k],s[k+1],got,want)
print('piece mismatches',bad)
print(repr(ask('PIECE 1 29871')), ask('PIECE 5 13'), ask('ENCODE '), ask('ENCODE'))
# FIXED argmax vs generated ids
pr=fx['prompts'][1]; seq=pr['seq']
r=ask('FIXED '+' '.join(map(str,seq))).split()
n0=len(seq)-len(pr['ids'])
print('FIXED consistent:', [int(x) for x in r[2+n0:]][:len(pr['ids'])-1]==pr['ids'][1:])
# decode timing
p.stdin.write('GENERATE 200 Once upon a time\n'); p.stdin.flush()
t0=time.time(); n=0
while True:
    l=p.stdout.readline()
    if l.startswith('T '): n+=1
    else: break
dt=time.time()-t0; print(l.strip(), '%.1f tok/s'%(n/dt))
# long FIXED: 1024 ids, timing; compare with chunk-size-independent sequential? compare first part with logits path
ids=[1]+[ (i*7919)%31000+500 for i in range(1023)]
t0=time.time(); r=ask('FIXED '+' '.join(map(str,ids))); print('FIXED 1024: %.3fs'%(time.time()-t0), r.split()[:3])
print(ask('FIXED 1 '+' '.join(['5']*1024))[:30])
# context stop: prompt near seq_len
p.stdin.write('GENERATE 2000 '+'hello '*700+'\n'); p.stdin.flush()
n=0
while True:
    l=p.stdout.readline()
    if l.startswith('T '): n+=1
    else: break
print(l.strip())
print(ask('GENERATE 0 hi'), ask('FIXED'), ask('FIXED 99999'))
p.stdin.write('QUIT\n'); p.stdin.flush(); print('exit', p.wait())
