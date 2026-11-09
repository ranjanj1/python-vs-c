import subprocess
p=subprocess.Popen(['./cb2','model.bin','tokenizer.bin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
p.stdout.readline()
def ask(s): p.stdin.write(s+'\n'); p.stdin.flush(); return p.stdout.readline().rstrip('\n')
for k in (1015, 1030):
    txt=' '.join(['dog']*k); n=len(ask('ENCODE '+txt).split())-1
    p.stdin.write(f'GENERATE 50 {txt}\n'); p.stdin.flush(); c=0
    while (l:=p.stdout.readline()).startswith('T '): c+=1
    print('prompt tokens',n,'->',l.strip(),'(expect',max(0,1024-n+1) if n<=1024 else 0,'if no stop)')
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait()
