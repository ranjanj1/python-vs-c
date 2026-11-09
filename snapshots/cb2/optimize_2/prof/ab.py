# alternate binaries; report median of per-run metrics
import subprocess, sys, statistics, re
bins=sys.argv[2:]; n=int(sys.argv[1]); res={b:[] for b in bins}
for i in range(n):
    for b in bins:
        out=subprocess.run(['python3','/tmp/cb2t/dev.py',b,'2'],capture_output=True,text=True).stdout
        res[b].append({k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',out)})
for b in bins:
    ks=res[b][0].keys()
    print(b.split('/')[-1], ' '.join(f"{k}={statistics.median(r[k] for r in res[b]):.1f}" for k in ks))
