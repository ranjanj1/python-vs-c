"""Exercise code paths the supplied model does not use, by comparing LOGITS with the normal build:
  1. unshared classifier: synthetic checkpoint = model.bin with negative vocab_size and the
     embedding table appended as an explicit classifier (must give identical logits);
  2. unpacked fallback: binary built with -DPR=3 (32000 % 3 != 0, so the classifier and the
     embedding lookup take the row-major path while 768-row matrices stay packed).
usage: python3 -I tests/paths.py <dir> <tmpdir>"""
import array, os, struct, subprocess, sys
d, t = sys.argv[1], sys.argv[2]
hdr = open(f"{d}/model.bin", "rb").read(28)
dim, hid, L, nh, nkv, V, S = struct.unpack("7i", hdr)
with open(f"{d}/model.bin", "rb") as f, open(f"{t}/unshared.bin", "wb") as g:
    f.read(28); g.write(struct.pack("7i", dim, hid, L, nh, nkv, -V, S))
    emb = f.read(V * dim * 4); g.write(emb)
    while True:
        b = f.read(1 << 24)
        if not b: break
        g.write(b)
    g.write(emb)
subprocess.run(["cc", "-O3", "-mcpu=apple-m1", "-std=c11", "-D_DARWIN_C_SOURCE", "-DPR=3", "-o", f"{t}/run_pr3", f"{d}/run_c1.c"], check=True)
seq = "1 9038 2501 263 931 29892 727 471 263 2217 7826 4257 365 2354 29889"
def logits(binary, model):
    p = subprocess.Popen([binary, model, f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline() == "READY\n"
    out = p.communicate(f"LOGITS {t}/l.f32 {seq}\nGENERATE 12 Once upon a time\nQUIT\n")[0].split("\n")
    a = array.array("f"); a.frombytes(open(f"{t}/l.f32", "rb").read()); return a, out[1:-1]
base, gb = logits(f"{d}/run_c1", f"{d}/model.bin")
for name, b, m in (("unshared checkpoint", f"{d}/run_c1", f"{t}/unshared.bin"), ("PR=3 fallback build", f"{t}/run_pr3", f"{d}/model.bin")):
    a, ga = logits(b, m)
    print(f"{name}: max |diff| vs normal = {max(abs(x - y) for x, y in zip(a, base)):.3g}, generate identical = {ga == gb}")
os.remove(f"{t}/unshared.bin")
