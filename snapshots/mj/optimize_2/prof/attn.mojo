# Attention microbenchmark (model shape: 12 heads x 64, kv_dim 768, cache layout (seq, kv_dim)).
from std.memory.alloc import alloc, Layout
from std.time import perf_counter_ns
from std.math import exp, sqrt
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime VW = 16
def fa(n: Int) -> FPtr:
    var p = alloc(Layout[Float32, alignment = .of_bytes[64]()](count=n)).unsafe_leak()
    for i in range(n):
        p[unsafe_offset=i] = Float32(((i * 2654435761) >> 7) % 1000) * 0.001 - 0.5
    return p

def softmax(x: FPtr, size: Int):
    var max_val = x[unsafe_offset=0]
    var i = 0
    if size >= VW:
        var mv = x.unsafe_load[width=VW]()
        i = VW
        while i + VW <= size:
            mv = max(mv, x.unsafe_offset(i).unsafe_load[width=VW]())
            i += VW
        max_val = mv.reduce_max()
    while i < size:
        if x[unsafe_offset=i] > max_val:
            max_val = x[unsafe_offset=i]
        i += 1
    var sv = SIMD[DType.float32, VW](0)
    i = 0
    while i + VW <= size:
        var e = exp(x.unsafe_offset(i).unsafe_load[width=VW]() - max_val)
        x.unsafe_offset(i).unsafe_store(e)
        sv += e
        i += VW
    var sum = sv.reduce_add()
    while i < size:
        var e = exp(x[unsafe_offset=i] - max_val)
        x[unsafe_offset=i] = e
        sum += e
        i += 1
    i = 0
    while i + VW <= size:
        x.unsafe_offset(i).unsafe_store(x.unsafe_offset(i).unsafe_load[width=VW]() / sum)
        i += VW
    while i < size:
        x[unsafe_offset=i] = x[unsafe_offset=i] / sum
        i += 1

# V1: current (position-major, all heads per row)
def v1[HS: Int](q: FPtr, att: FPtr, kb: FPtr, vb: FPtr, kv_dim: Int, seq_len: Int, npos: Int, xb: FPtr, sqrt_hs: Float32, n_heads: Int, kv_mul: Int):
    for t in range(npos):
        var krow = kb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            var qv = q.unsafe_offset(h * HS).unsafe_load[width=HS]()
            var kv = krow.unsafe_offset((h // kv_mul) * HS).unsafe_load[width=HS]()
            att[unsafe_offset=h * seq_len + t] = (qv * kv).reduce_add() / sqrt_hs
    for h in range(n_heads):
        softmax(att.unsafe_offset(h * seq_len), npos)
        xb.unsafe_offset(h * HS).unsafe_store(SIMD[DType.float32, HS](0))
    for t in range(npos):
        var vrow = vb.unsafe_offset(t * kv_dim)
        for h in range(n_heads):
            var a = att[unsafe_offset=h * seq_len + t]
            var o = xb.unsafe_offset(h * HS)
            o.unsafe_store(vrow.unsafe_offset((h // kv_mul) * HS).unsafe_load[width=HS]().fma(SIMD[DType.float32, HS](a), o.unsafe_load[width=HS]()))

# V2: tiles of TP positions; per head, q and the output accumulator stay in registers over the tile;
# no integer division in the inner loops.
def v2[HS: Int, TP: Int](q: FPtr, att: FPtr, kb: FPtr, vb: FPtr, kv_dim: Int, seq_len: Int, npos: Int, xb: FPtr, sqrt_hs: Float32, n_heads: Int, kv_mul: Int):
    for h in range(n_heads):
        var qv = q.unsafe_offset(h * HS).unsafe_load[width=HS]()
        var kh = kb.unsafe_offset((h // kv_mul) * HS)
        var ah = att.unsafe_offset(h * seq_len)
        for t in range(npos):
            ah[unsafe_offset=t] = (qv * kh.unsafe_offset(t * kv_dim).unsafe_load[width=HS]()).reduce_add() / sqrt_hs
    for h in range(n_heads):
        softmax(att.unsafe_offset(h * seq_len), npos)
    for h in range(n_heads):
        var vh = vb.unsafe_offset((h // kv_mul) * HS)
        var ah = att.unsafe_offset(h * seq_len)
        var acc = SIMD[DType.float32, HS](0)
        for t in range(npos):
            acc = vh.unsafe_offset(t * kv_dim).unsafe_load[width=HS]().fma(SIMD[DType.float32, HS](ah[unsafe_offset=t]), acc)
        xb.unsafe_offset(h * HS).unsafe_store(acc)

# V3: like V1 (position-major) but scores for 4 positions per inner step and hoisted head offsets
def v3[HS: Int](q: FPtr, att: FPtr, kb: FPtr, vb: FPtr, kv_dim: Int, seq_len: Int, npos: Int, xb: FPtr, sqrt_hs: Float32, n_heads: Int, kv_mul: Int):
    for h in range(n_heads):
        var qv = q.unsafe_offset(h * HS).unsafe_load[width=HS]()
        var kh = kb.unsafe_offset((h // kv_mul) * HS)
        var ah = att.unsafe_offset(h * seq_len)
        var t = 0
        while t + 4 <= npos:
            var s0 = (qv * kh.unsafe_offset(t * kv_dim).unsafe_load[width=HS]()).reduce_add[4]()
            var s1 = (qv * kh.unsafe_offset((t + 1) * kv_dim).unsafe_load[width=HS]()).reduce_add[4]()
            var s2 = (qv * kh.unsafe_offset((t + 2) * kv_dim).unsafe_load[width=HS]()).reduce_add[4]()
            var s3 = (qv * kh.unsafe_offset((t + 3) * kv_dim).unsafe_load[width=HS]()).reduce_add[4]()
            ah[unsafe_offset=t] = s0.reduce_add() / sqrt_hs
            ah[unsafe_offset=t + 1] = s1.reduce_add() / sqrt_hs
            ah[unsafe_offset=t + 2] = s2.reduce_add() / sqrt_hs
            ah[unsafe_offset=t + 3] = s3.reduce_add() / sqrt_hs
            t += 4
        while t < npos:
            ah[unsafe_offset=t] = (qv * kh.unsafe_offset(t * kv_dim).unsafe_load[width=HS]()).reduce_add() / sqrt_hs
            t += 1
    for h in range(n_heads):
        softmax(att.unsafe_offset(h * seq_len), npos)
    for i in range(n_heads * HS):
        xb[unsafe_offset=i] = 0.0
    for t in range(npos):
        var vrow = vb.unsafe_offset(t * kv_dim)
        var arow = att.unsafe_offset(t)
        for h in range(n_heads):
            var o = xb.unsafe_offset(h * HS)
            o.unsafe_store(vrow.unsafe_offset(h * HS).unsafe_load[width=HS]().fma(SIMD[DType.float32, HS](arow[unsafe_offset=h * seq_len]), o.unsafe_load[width=HS]()))

def bench[which: Int](npos: Int, name: StaticString):
    comptime L = 12
    var seq = 1024
    var kvd = 768
    var kc = fa(L * seq * kvd)
    var vc = fa(L * seq * kvd)
    var q = fa(768)
    var att = fa(12 * seq)
    var xb = fa(768)
    var best = Float64(1e18)
    var chk: Float32 = 0
    for rep in range(15):
        var t0 = perf_counter_ns()
        for l in range(L):
            var kb = kc.unsafe_offset(l * seq * kvd)
            var vb = vc.unsafe_offset(l * seq * kvd)
            comptime if which == 1:
                v1[64](q, att, kb, vb, kvd, seq, npos, xb, 8.0, 12, 1)
            elif which == 2:
                v2[64, 8](q, att, kb, vb, kvd, seq, npos, xb, 8.0, 12, 1)
            else:
                v3[64](q, att, kb, vb, kvd, seq, npos, xb, 8.0, 12, 1)
            chk += xb[unsafe_offset=5]
        var dt = Float64(perf_counter_ns() - t0)
        if dt < best: best = dt
    print(name, "npos", npos, "ms/token (12 layers)", best / 1e6, chk)

def main():
    for np in [64, 512, 1024]:
        bench[1](np, "V1 current        ")
        bench[2](np, "V2 per-head regs  ")
        bench[3](np, "V3 4-pos scores   ")
