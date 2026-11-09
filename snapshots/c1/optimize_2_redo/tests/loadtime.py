"""Time from process start to READY (median of N). usage: python3 -I tests/loadtime.py <dir> <cmd...>"""
import subprocess, sys, time, statistics
d = sys.argv[1]; cmd = sys.argv[2:]; ts = []
for _ in range(5):
    t = time.time()
    p = subprocess.Popen(cmd + [f"{d}/model.bin", f"{d}/tokenizer.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline() == "READY\n"; ts.append(time.time() - t)
    p.communicate("QUIT\n")
print(f"READY after median {1000*statistics.median(ts):.0f} ms (min {1000*min(ts):.0f}, max {1000*max(ts):.0f})")
