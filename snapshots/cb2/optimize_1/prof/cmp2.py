import subprocess, array
ids=[1]+[(i*7919)%31000+500 for i in range(599)]
def run(exe,tag):
    p=subprocess.Popen([exe,'model.bin','tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    p.stdout.readline(); f=f'/tmp/cb2t/{tag}.f32'
    p.stdin.write(f'LOGITS {f} '+' '.join(map(str,ids))+'\nQUIT\n'); p.stdin.flush(); p.stdout.readline(); p.wait()
    a=array.array('f'); a.frombytes(open(f,'rb').read()); return a
a=run('/tmp/cb2t/c1_v1','v1seq'); b=run('/tmp/cb2t/c1','v2seq'); c=run('./cb2','v2chunk')
print('new sequential vs v1 sequential:', max(abs(x-y) for x,y in zip(a,b)))
print('new chunked    vs v1 sequential:', max(abs(x-y) for x,y in zip(a,c)))
