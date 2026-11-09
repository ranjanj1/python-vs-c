import sys, time
sys.path.insert(0, sys.argv[1])
from run import Tokenizer
t=time.perf_counter(); tk=Tokenizer('tokenizer.bin',32000); print('tok load %.3fs'%(time.perf_counter()-t))
for txt in ['Once upon a time', 'The little dog was sad because he could not find his red ball anywhere in the house', ' '.join(['cat']*1020), 'Lily went to the park and played with her friends. '*60]:
    b=txt.encode(); t=time.perf_counter(); ids=tk.encode(b); print(len(b),'bytes',len(ids),'ids %.2f ms'%((time.perf_counter()-t)*1e3))
