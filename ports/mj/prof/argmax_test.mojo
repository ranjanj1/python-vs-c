from std.memory.alloc import alloc
from std.random import random_float64, random_ui64, seed
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]

def argmax_vec(lg: FPtr, n: Int) -> Int:
    # Same result as run.c's sample_argmax: the first index of the largest
    # value, where only a strictly greater value replaces the running max
    # (seeded with logits[0], so NaNs are never selected after index 0).
    # Pass 1 finds that max value with lane-wise `>`-selects (vectorized);
    # pass 2 returns the first index holding it.
    var m0 = lg[unsafe_offset=0]
    if m0 != m0:  # logits[0] is NaN: nothing compares greater, run.c returns 0
        return 0
    comptime W = 16
    var mv = SIMD[DType.float32, W](m0)
    var i = 0
    while i + W <= n:
        var v = lg.unsafe_offset(i).unsafe_load[width=W]()
        mv = v.gt(mv).select(v, mv)
        i += W
    var max_p = m0
    for k in range(W):
        if mv[k] > max_p:
            max_p = mv[k]
    while i < n:
        if lg[unsafe_offset=i] > max_p:
            max_p = lg[unsafe_offset=i]
        i += 1
    # first index equal to the maximum (+0.0 == -0.0, as with `>` in run.c)
    i = 0
    while i + W <= n:
        if lg.unsafe_offset(i).unsafe_load[width=W]().eq(max_p).reduce_or():
            break
        i += W
    while i < n:
        if lg[unsafe_offset=i] == max_p:
            return i
        i += 1
    return 0



def argmax_ref(p: FPtr, n: Int) -> Int:  # run.c sample_argmax
    var max_i = 0
    var max_p = p[unsafe_offset=0]
    for i in range(1, n):
        if p[unsafe_offset=i] > max_p:
            max_i = i
            max_p = p[unsafe_offset=i]
    return max_i

def main():
    seed(1234)
    var buf = alloc[Float32]({count = 40000}).unsafe_leak()
    var nan = Float32(0) / Float32(0)
    var inf = Float32(1) / Float32(0)
    var bad = 0
    for trial in range(20000):
        var n = Int(random_ui64(1, 33000)) if trial % 3 else 32000
        var mode = trial % 7
        for i in range(n):
            var r = random_float64()
            var v: Float32
            if mode == 0:
                v = Float32(Int(r * 4))           # many exact ties
            elif mode == 1:
                v = -0.0 if r < 0.5 else 0.0      # signed zeros only
            elif mode == 2:
                v = nan if r < 0.1 else Float32(r * 10 - 5)
            elif mode == 3:
                v = -inf if r < 0.5 else Float32(r - 2)
            else:
                v = Float32(r * 30 - 15)
            buf[unsafe_offset=i] = v
        if trial % 11 == 0: buf[unsafe_offset=0] = nan
        if trial % 13 == 0: buf[unsafe_offset=n - 1] = inf    # max in the tail
        if trial % 17 == 0: buf[unsafe_offset=0] = inf
        if argmax_vec(buf, n) != argmax_ref(buf, n):
            bad += 1
            if bad < 5: print("MISMATCH trial", trial, "n", n, argmax_vec(buf, n), argmax_ref(buf, n))
    print("argmax trials 20000, mismatches", bad)
