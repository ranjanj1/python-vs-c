from std.memory.alloc import alloc, Layout
from std.time import perf_counter_ns
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
def sweep[S: Int](p: FPtr, n: Int) -> Float64:
    # S sequential streams, each over a contiguous 1/S of the buffer
    var best = Float64(1e18)
    var seg = n // S
    var tot: Float32 = 0
    for rep in range(6):
        var t0 = perf_counter_ns()
        var acc = Array[SIMD[DType.float32, 16], S](fill=0)
        var j = 0
        while j + 16 <= seg:
            comptime for s in range(S):
                acc[s] += p.unsafe_offset(s * seg + j).unsafe_load[width=16]()
            j += 16
        comptime for s in range(S):
            tot += acc[s].reduce_add()
        var dt = Float64(perf_counter_ns() - t0)
        if dt < best: best = dt
    print(S, "streams GB/s", Float64(n * 4) / best, tot)
    return best
def main():
    var n = 438381596 // 4
    var p = alloc(Layout[Float32, alignment = .of_bytes[64]()](count=n)).unsafe_leak()
    for i in range(n):
        p[unsafe_offset=i] = Float32(i & 7)
    _ = sweep[1](p, n); _ = sweep[2](p, n); _ = sweep[4](p, n); _ = sweep[8](p, n); _ = sweep[1](p, n)
