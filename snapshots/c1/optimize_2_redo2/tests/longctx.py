"""Compare run_c1 LOGITS with the unmodified reference over a full-context sequence,
and check GENERATE context/stop behaviour. usage: python3 -I tests/longctx.py <dir> [n]"""
import array, random, subprocess, sys, tempfile, time
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024; V = 32000
random.seed(7)
seq = [1] + [random.randrange(3, V) for _ in range(n - 1)]
with tempfile.TemporaryDirectory() as t:
    t0 = time.time()
    subprocess.run([f"{d}/tests/ref_logits", f"{d}/model.bin", f"{t}/r.f32"] + list(map(str, seq)), check=True)
    t1 = time.time()
    p = subprocess.Popen([f"{d}/run_c1", f"{d}/model.bin", f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline() == "READY\n"
    t2 = time.time()
    p.stdin.write(f"LOGITS {t}/c.f32 " + " ".join(map(str, seq)) + "\n"); p.stdin.flush()
    print(p.stdout.readline().strip()[:40]); t3 = time.time()
    p.stdin.write("FIXED " + " ".join(map(str, seq)) + "\n"); p.stdin.flush()
    fixed = p.stdout.readline().split()
    a = array.array('f'); a.frombytes(open(f"{t}/r.f32", "rb").read())
    b = array.array('f'); b.frombytes(open(f"{t}/c.f32", "rb").read())
    assert len(a) == len(b) == n * V
    worst = max(max(abs(x - y) for x, y in zip(a[i*V:(i+1)*V], b[i*V:(i+1)*V])) for i in range(0, n, 1))
    am = [max(range(V), key=lambda j: (a[i*V+j], -j)) for i in range(n)]
    print(f"positions {n}: worst max-abs logit error {worst:.3g}; argmax agree {sum(int(x)==y for x,y in zip(fixed[2:], am))}/{n}")
    print(f"reference {t1-t0:.1f}s, run_c1 LOGITS {t3-t2:.2f}s")
    # context: long prompt so that generation runs into seq_len
    p.stdin.write("GENERATE 2000 Once upon a time\n"); p.stdin.flush()
    cnt = 0
    while True:
        s = p.stdout.readline().strip()
        if not s.startswith("T "): break
        cnt += 1
    print("GENERATE 2000:", s, "tokens", cnt)
    p.stdin.write("GENERATE 0 hi\nQUIT\n"); p.stdin.flush()
    print("GENERATE 0:", p.stdout.readline().strip()); p.wait(); print("exit", p.returncode)
