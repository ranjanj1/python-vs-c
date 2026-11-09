"""Rough local timing (the official timing is the harness's). usage: python3 -I tests/bench.py <dir> <cmd...>"""
import subprocess, sys, time
d = sys.argv[1]; cmd = sys.argv[2:]
t0 = time.time()
p = subprocess.Popen(cmd + [f"{d}/model.bin", f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
assert p.stdout.readline() == "READY\n"; t1 = time.time()
def gen(n, text):
    p.stdin.write(f"GENERATE {n} {text}\n"); p.stdin.flush(); c = 0; t = time.time(); first = None
    while True:
        s = p.stdout.readline()
        if first is None: first = time.time() - t
        if not s.startswith("T "): return c, time.time() - t, first, s.strip()
        c += 1
print(f"load {t1-t0:.2f}s")
for n in (16, 128, 256):
    c, dt, first, s = gen(n, "Once upon a time")
    print(f"GENERATE {n}: {s}, {dt:.2f}s, {1000*dt/(c+4):.2f} ms/forward, first token {1000*first:.1f}ms")
ids = " ".join(["1"] + ["450"] * 511)
t = time.time(); p.stdin.write(f"FIXED {ids}\n"); p.stdin.flush(); p.stdout.readline()
print(f"FIXED 512: {time.time()-t:.2f}s")
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait()
