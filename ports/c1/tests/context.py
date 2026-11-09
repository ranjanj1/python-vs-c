"""Exercise the 'context' stop: long prompt + GENERATE until seq_len. usage: python3 -I tests/context.py <dir>"""
import subprocess, sys
d = sys.argv[1]
p = subprocess.Popen([f"{d}/run_c1", f"{d}/model.bin", f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
assert p.stdout.readline() == "READY\n"
text = " ".join(["one two three four five six seven eight nine ten"] * 120)
for words in (980, 1005, 1100):
    t = " ".join(text.split()[:words])
    p.stdin.write(f"ENCODE {t}\n"); p.stdin.flush(); n = len(p.stdout.readline().split()) - 1
    p.stdin.write(f"GENERATE 500 {t}\n"); p.stdin.flush(); c = 0
    while True:
        s = p.stdout.readline().strip()
        if not s.startswith("T "): break
        c += 1
    print(f"prompt tokens {n}: {s} (expected 'DONE {max(0, 1024 - n + 1)} context' unless stop)")
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait()
