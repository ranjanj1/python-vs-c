# Development workload driver: replays the supplied fixture workloads and times each phase from outside.
import json, subprocess, time, sys, tempfile, os
exe=sys.argv[1]; reps=int(sys.argv[2]) if len(sys.argv)>2 else 3
fx=json.load(open('/Users/agentlab/pvc/cb2/fixtures.json'))
t0=time.perf_counter()
p=subprocess.Popen([exe,'/Users/agentlab/pvc/cb2/model.bin','/Users/agentlab/pvc/cb2/tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
assert p.stdout.readline().strip()=='READY'; load=time.perf_counter()-t0
def ask(s): p.stdin.write(s+'\n'); p.stdin.flush(); return p.stdout.readline().rstrip('\n')
res={'load':load}
def best(name,f):
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); f(); b=min(b,time.perf_counter()-t)
    res[name]=b
def enc():
    for t in fx['tokenizer']: ask('ENCODE '+t)
best('encode_all',enc)
def gen():
    for pr in fx['prompts']:
        p.stdin.write(f"GENERATE 64 {pr['text']}\n"); p.stdin.flush(); got=[]
        while (l:=p.stdout.readline()).startswith('T '): got.append(int(l[2:]))
        assert got==pr['ids'][:64], (got, l)
best('generate_3x64',gen)
d=tempfile.mkdtemp()
def lg():
    for pr in fx['prompts']: ask(f"LOGITS {d}/l.f32 "+' '.join(map(str,pr['seq'])))
best('logits_3',lg)
def fixed():
    for pr in fx['prompts']: ask("FIXED "+' '.join(map(str,pr['seq'])))
best('fixed_3',fixed)
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
print(os.path.basename(exe), ' '.join(f'{k}={v*1000:.1f}ms' for k,v in res.items()), f"gen_tok/s={192/res['generate_3x64']:.1f}")
