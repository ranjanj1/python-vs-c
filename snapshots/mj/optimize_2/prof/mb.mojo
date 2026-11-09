# Microbenchmark: matvec over the model's real weight shapes, row-major (R rows = R streams)
# vs packed (R rows interleaved in W-wide chunks = one sequential stream).
from std.memory.alloc import alloc, Layout
from std.time import perf_counter_ns
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]

def fa(n: Int) -> FPtr:
    var p = alloc(Layout[Float32, alignment = .of_bytes[64]()](count=n)).unsafe_leak()
    for i in range(n):
        p[unsafe_offset=i] = Float32((i * 7) % 13) * 0.01
    return p

def mm_rows[R: Int, W: Int](xout: FPtr, x: FPtr, w: FPtr, n: Int, d: Int):
    var i = 0
    while i + R <= d:
        var rb = w.unsafe_offset(i * n)
        var acc = Array[SIMD[DType.float32, W], R](fill=0)
        var j = 0
        while j + W <= n:
            var xv = x.unsafe_offset(j).unsafe_load[width=W]()
            comptime for r in range(R):
                acc[r] = rb.unsafe_offset(r * n + j).unsafe_load[width=W]().fma(xv, acc[r])
            j += W
        comptime for r in range(R):
            xout[unsafe_offset=i + r] = acc[r].reduce_add()
        i += R

def mm_packed[R: Int, W: Int](xout: FPtr, x: FPtr, w: FPtr, n: Int, d: Int):
    var i = 0
    var p = w
    while i + R <= d:
        var acc = Array[SIMD[DType.float32, W], R](fill=0)
        var j = 0
        while j + W <= n:
            var xv = x.unsafe_offset(j).unsafe_load[width=W]()
            comptime for r in range(R):
                acc[r] = p.unsafe_offset(r * W).unsafe_load[width=W]().fma(xv, acc[r])
            p = p.unsafe_offset(R * W)
            j += W
        comptime for r in range(R):
            xout[unsafe_offset=i + r] = acc[r].reduce_add()
        i += R

def run[packed: Bool, R: Int, W: Int](w: FPtr, x: FPtr, y: FPtr, name: StaticString):
    var best = Float64(1e18)
    for rep in range(6):
        var t0 = perf_counter_ns()
        var off = 0
        for l in range(12):
            comptime for m in range(7):
                comptime n = 2048 if m == 6 else 768
                comptime d = 768 if (m < 4 or m == 6) else 2048
                comptime if packed:
                    mm_packed[R, W](y, x, w.unsafe_offset(off), n, d)
                else:
                    mm_rows[R, W](y, x, w.unsafe_offset(off), n, d)
                off += n * d
        comptime if packed:
            mm_packed[R, W](y, x, w.unsafe_offset(off), 768, 32000)
        else:
            mm_rows[R, W](y, x, w.unsafe_offset(off), 768, 32000)
        off += 768 * 32000
        var dt = Float64(perf_counter_ns() - t0)
        if dt < best: best = dt
    var bytes = Float64(12 * (4 * 768 * 768 + 3 * 768 * 2048) + 768 * 32000) * 4
    print(name, "ms", best / 1e6, "GB/s", bytes / best)

def main():
    var total = 12 * (4 * 768 * 768 + 3 * 768 * 2048) + 768 * 32000
    var w = fa(total)
    var x = fa(2048)
    var y = fa(32000)
    run[False, 2, 32](w, x, y, "rows   R=2 W=32")
    run[False, 4, 16](w, x, y, "rows   R=4 W=16")
    run[True, 2, 32](w, x, y, "packed R=2 W=32")
    run[True, 4, 16](w, x, y, "packed R=4 W=16")
    run[True, 4, 32](w, x, y, "packed R=4 W=32")
    run[True, 8, 16](w, x, y, "packed R=8 W=16")
    run[True, 8, 8](w, x, y, "packed R=8 W=8")
    run[True, 16, 8](w, x, y, "packed R=16 W=8")
    run[False, 2, 32](w, x, y, "rows   R=2 W=32")
