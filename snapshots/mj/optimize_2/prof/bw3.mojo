# Streaming-read ceiling: plain loads vs non-temporal loads vs software prefetch at various distances.
from std.memory.alloc import alloc, Layout
from std.time import perf_counter_ns
from std.sys.intrinsics import prefetch
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
def sweep[NT: Bool, PF: Int](p: FPtr, n: Int, name: StaticString):
    var best = Float64(1e18)
    var tot: Float32 = 0
    for rep in range(7):
        var t0 = perf_counter_ns()
        var a0 = SIMD[DType.float32, 16](0)
        var a1 = SIMD[DType.float32, 16](0)
        var j = 0
        while j + 32 <= n:
            comptime if PF > 0:
                if j % 64 == 0:
                    prefetch(p.unsafe_offset(j + PF))
            a0 += p.unsafe_offset(j).unsafe_load[width=16, non_temporal=NT]()
            a1 += p.unsafe_offset(j + 16).unsafe_load[width=16, non_temporal=NT]()
            j += 32
        tot += (a0 + a1).reduce_add()
        var dt = Float64(perf_counter_ns() - t0)
        if dt < best: best = dt
    print(name, "GB/s", Float64(n * 4) / best, tot)
def main():
    var n = 438381596 // 4
    var p = alloc(Layout[Float32, alignment = .of_bytes[64]()](count=n + 65536)).unsafe_leak()
    for i in range(n + 65536):
        p[unsafe_offset=i] = Float32(i & 7)
    sweep[False, 0](p, n, "plain          ")
    sweep[True, 0](p, n, "non-temporal   ")
    sweep[False, 1024](p, n, "prefetch 4KB   ")
    sweep[False, 4096](p, n, "prefetch 16KB  ")
    sweep[False, 8192](p, n, "prefetch 32KB  ")
    sweep[False, 16384](p, n, "prefetch 64KB  ")
    sweep[False, 0](p, n, "plain          ")
