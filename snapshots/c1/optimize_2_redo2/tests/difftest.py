"""Differential test: run_c1 vs the unmodified reference for ENCODE and PIECE.
usage: python3 -I tests/difftest.py <repo_dir>"""
import random, subprocess, sys
d = sys.argv[1]
random.seed(1)
alpha = list("abcdefghijklmnopqrstuvwxyz ABCXYZ.,!?'\"\t0123456789-") + ["é", "ï", "日", "本", "語", "😀", "ß", " ", "  "]
texts = ["", " ", "  ", "a", "Once upon a time", "\x7f", "\xc3", "abc\x80def"]
for _ in range(2000):
    texts.append("".join(random.choice(alpha) for _ in range(random.randint(1, 60))))
texts.append("".join(random.choice(alpha) for _ in range(3000)))
texts = [t for t in texts if "\n" not in t]
raw = ("\n".join(texts) + "\n").encode("utf-8", "surrogateescape")
ref = subprocess.run([f"{d}/tests/ref_encode", f"{d}/tokenizer.bin"], input=raw, capture_output=True).stdout.decode().splitlines()
p = subprocess.Popen([f"{d}/run_c1", f"{d}/model.bin", f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
assert p.stdout.readline() == b"READY\n"
cmds = [b"ENCODE " + t.encode("utf-8", "surrogateescape") for t in texts]
cmds += [f"PIECE {prev} {i}".encode() for prev in range(3) for i in range(32000)]
got = p.communicate(b"\n".join(cmds) + b"\nQUIT\n")[0].decode().splitlines()
bad = sum(1 for a, b in zip(ref, got) if a != b) + abs(len(ref) - len(got))
print(f"{len(ref)} answers compared, {bad} mismatches")
for (i, (a, b)) in enumerate(zip(ref, got)):
    if a != b:
        print(i, cmds[i][:80], a[:80], b[:80]); break
sys.exit(1 if bad else 0)
