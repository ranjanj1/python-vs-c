import subprocess,time,json,sys
fx=json.load(open('/Users/agentlab/pvc/cb2/fixtures.json'))
for exe in sys.argv[1:]:
    p=subprocess.Popen([exe,'/Users/agentlab/pvc/cb2/model.bin','/Users/agentlab/pvc/cb2/tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True); p.stdout.readline()
    out=[]
    for pr in fx['prompts']:
        best=(9,9)
        for _ in range(3):
            t=time.perf_counter(); p.stdin.write(f"GENERATE 16 {pr['text']}\n"); p.stdin.flush()
            p.stdout.readline(); t1=time.perf_counter()-t
            while p.stdout.readline().startswith('T '): pass
            t2=time.perf_counter()-t; best=min(best,(t1,(t2-t1)/15))
        n=len(subprocess.os.popen('true').read())  # noop
        out.append(f"TTFT {best[0]*1e3:.1f} ms, step {best[1]*1e3:.2f} ms")
    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
    print(exe.split('/')[-1], ' | '.join(out))
