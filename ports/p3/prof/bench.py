import sys, time, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import importlib; _m = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "run"); Transformer, Tokenizer = _m.Transformer, _m.Tokenizer
t0=time.perf_counter(); m=Transformer('model.bin'); tk=Tokenizer('tokenizer.bin', m.vocab_size); print('load %.3fs'%(time.perf_counter()-t0))
fx=json.load(open('fixtures.json'))
for pr in fx['prompts']:
    seq=pr['seq']; m.reset()
    t=time.perf_counter()
    for p,tok in enumerate(seq): m.forward(tok,p)
    dt=time.perf_counter()-t; print('%d positions: %.2f ms/token'%(len(seq),dt/len(seq)*1e3))
m.reset(); t=time.perf_counter()
for p in range(1024): m.forward(5,p)
print('1024 positions: %.2f ms/token'%((time.perf_counter()-t)/1024*1e3))
