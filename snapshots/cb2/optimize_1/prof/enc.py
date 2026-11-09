# compare ENCODE of new build against v1 (verbatim reference algorithm) on randomized texts
import subprocess, random, json, time
def start(exe):
    p=subprocess.Popen([exe,'model.bin','tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',errors='surrogateescape'); p.stdout.readline(); return p
A=start('/tmp/cb2t/v1'); B=start('./cb2')
random.seed(7)
words=['the','a','Lily',' ','  ','dog','ball','!','?','12345','naïve','café','日本語','😀','\t','ünïcödé','x'*30,'aaaa','<0x41>','ab','ba','...',',',' ','Ω','​']
fx=json.load(open('fixtures.json'))
texts=list(fx['tokenizer'])+[p['text'] for p in fx['prompts']]
for _ in range(400):
    texts.append(''.join(random.choice(words)+random.choice(['',' ']) for _ in range(random.randint(1,60))))
for _ in range(100):
    texts.append(''.join(chr(random.choice([random.randint(32,126),random.randint(160,0x7ff),random.randint(0x800,0xffff) if random.random()<.5 else 65,random.randint(0x10000,0x1f6ff)])) for _ in range(random.randint(1,80))).replace('\ud800','a'))
texts.append(' '.join(['dog']*900)); texts.append('Once upon a time, there was a little girl named Lily. '*40)
bad=0; ta=tb=0
for t in texts:
    if any(0xd800<=ord(ch)<=0xdfff for ch in t) or '\n' in t or '\r' in t: continue
    for P in (A,B): P.stdin.write('ENCODE '+t+'\n'); P.stdin.flush()
    t0=time.perf_counter(); ra=A.stdout.readline(); t1=time.perf_counter(); rb=B.stdout.readline(); t2=time.perf_counter()
    if ra!=rb: bad+=1; print('MISMATCH',repr(t[:60]))
print(len(texts),'texts, mismatches:',bad)
for P in (A,B): P.stdin.write('QUIT\n'); P.stdin.flush(); P.wait()
