# Compare ./cb2 LOGITS and ENCODE against the supplied reference (prof/refdrv.c built at /tmp/cb2t/refdrv).
import subprocess, array, random, json, sys
REF='/tmp/cb2t/refdrv'; M='model.bin'; TK='tokenizer.bin'
n=int(sys.argv[1]) if len(sys.argv)>1 else 300
ids=[1]+[(i*7919)%31000+500 for i in range(n-1)]
subprocess.run([REF,M,TK,'logits','/tmp/cb2t/ref.f32']+list(map(str,ids)),check=True)
p=subprocess.Popen(['./cb2',M,TK],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True); p.stdout.readline()
p.stdin.write('LOGITS /tmp/cb2t/cb2.f32 '+' '.join(map(str,ids))+'\n'); p.stdin.flush(); print(p.stdout.readline().strip())
a=array.array('f'); a.frombytes(open('/tmp/cb2t/ref.f32','rb').read())
b=array.array('f'); b.frombytes(open('/tmp/cb2t/cb2.f32','rb').read())
V=len(a)//n; worst=max(max(abs(a[i]-b[i]) for i in range(r*V,(r+1)*V)) for r in range(n))
am_a=[max(range(V),key=lambda j:a[r*V+j]) for r in range(n)]; am_b=[max(range(V),key=lambda j:b[r*V+j]) for r in range(n)]
print(f'LOGITS vs reference over {n} positions: max abs diff {worst:.3g}, argmax agreement {sum(x==y for x,y in zip(am_a,am_b))}/{n}')
random.seed(7)
words=['the','a','Lily',' ','  ','dog','ball','!','?','12345','naïve','café','日本語','😀','\t','ünïcödé','x'*30,'aaaa','<0x41>','ab','...',',','Ω']
fx=json.load(open('fixtures.json'))
texts=[t for t in fx['tokenizer'] if t]+[q['text'] for q in fx['prompts']]
texts+=[''.join(random.choice(words)+random.choice(['',' ']) for _ in range(random.randint(1,60))) for _ in range(300)]
texts+=[' '.join(['dog']*900), 'Once upon a time, there was a little girl named Lily. '*40]
r=subprocess.run([REF,M,TK,'encode'],input=''.join(t+'\n' for t in texts),capture_output=True,text=True).stdout.splitlines()
bad=0
for t,want in zip(texts,r):
    p.stdin.write('ENCODE '+t+'\n'); p.stdin.flush()
    if p.stdout.readline().rstrip('\n')!=want: bad+=1; print('MISMATCH',repr(t[:50]))
print(f'ENCODE vs reference: {len(texts)} texts, {bad} mismatches')
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
