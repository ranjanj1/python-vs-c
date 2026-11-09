# Load strategies for building the column-blocked weights.
import time, struct, numpy as np, sys
sys.path.insert(0, '.')
from run import blocked, block_size
F = np.float32; pc = time.perf_counter
path = 'model.bin'
dim, hid, L, H, KVH, V, S = struct.unpack('<7i', open(path, 'rb').read(28)); V = abs(V)
mats = [('emb', 1, V, dim)] + [('rms', 0, L, dim)] + [(k, 1, L * a, b) for k, a, b in (('wq', dim, dim), ('wk', dim, dim), ('wv', dim, dim), ('wo', dim, dim))] + [('rms', 0, L, dim), ('w1', 1, L * hid, dim), ('w2', 1, L * dim, hid), ('w3', 1, L * hid, dim)]
def A():  # memmap + ascontiguousarray of a moved-axis view
    d = np.memmap(path, dtype='<f4', mode='r', offset=28); off = 0; out = []
    for name, blk, r, c in mats:
        v = d[off:off + r * c].reshape(r, c); off += r * c
        out.append(blocked(v, block_size(c)) if blk else np.array(v))
    return out
def B():  # sequential readinto of ~4 MB chunks into a scratch buffer, then copy into the blocked destination
    out = []; f = open(path, 'rb', buffering=0); f.seek(28)
    scratch = np.empty(4 * 1024 * 1024 // 4, F)
    for name, blk, r, c in mats:
        if not blk:
            a = np.empty((r, c), F); f.readinto(a); out.append(a); continue
        b = block_size(c); nb = c // b
        dest = np.empty((nb, r, b), F)
        rows = max(1, scratch.size // c)
        for i in range(0, r, rows):
            n = min(rows, r - i); buf = scratch[:n * c]; f.readinto(buf)
            dest[:, i:i + n, :] = buf.reshape(n, nb, b).transpose(1, 0, 2)
        out.append(dest)
    return out
for name, fn in (('A memmap+copy', A), ('B readinto chunks', B), ('A memmap+copy', A), ('B readinto chunks', B)):
    t = pc(); r = fn(); print('%-18s %.3fs' % (name, pc() - t)); del r
t = pc(); a = np.fromfile(path, dtype=F, offset=28); print('plain fromfile     %.3fs' % (pc() - t))
