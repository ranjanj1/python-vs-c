import subprocess,time,sys
for exe in sys.argv[1:]:
    p=subprocess.Popen([exe,'model.bin','tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True); p.stdout.readline()
    best=9e9
    for _ in range(2):
        p.stdin.write('GENERATE 100 '+' '.join(['dog']*900)+'\n'); p.stdin.flush(); t=time.perf_counter()
        while p.stdout.readline().startswith('T '): pass
        best=min(best,time.perf_counter()-t)
    t=time.perf_counter(); p.stdin.write('FIXED '+' '.join(['1']+[str((i*7919)%31000+500) for i in range(1023)])+'\n'); p.stdin.flush(); p.stdout.readline(); f=time.perf_counter()-t
    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
    print(exe.split('/')[-1], f'GENERATE 100 after 901-token prompt: {best*1000:.0f} ms   FIXED 1024: {f*1000:.0f} ms')
