# P1 — pure Python port of reference/run.c

Implementation: `run.py` (Python 3.12 standard library only: `sys`, `math`, `struct`, `array`).
Start: `python3 run.py <checkpoint.bin> <tokenizer.bin>`, then the INTERFACE.md line protocol.
Validation: `python3 check.py -- python3 -I run.py`. The checker appends the model and
tokenizer paths itself.

I did not read check.py, fixtures.json or fixtures.f32 (RULES.md 6). I read only RULES.md,
INTERFACE.md and reference/run.c. I did not look at any other port or external code.
Scratch benchmark scripts live in /tmp/p1bench (outside the project folder).

## Commands run

Port (session 1):
```
python3 --version                      # Python 3.12.13 (/Users/Shared/pvc-kit/stdlib/bin/python3)
python3 -c '...struct.unpack("7i")...' # model header: dim 768, hidden 2048, 12 layers, 12 heads,
                                       # 12 kv heads, vocab 32000 (shared classifier), seq_len 1024
python3 check.py                       # only to learn the invocation (needs `-- <program>`)
python3 -I /tmp/p1bench/b.py           # micro-benchmark of pure-Python dot-product forms
printf 'ENCODE ...\nPIECE ...\nFIXED ...\nGENERATE ...\nQUIT\n' | python3 -I run.py model.bin tokenizer.bin
python3 check.py -- python3 -I run.py model.bin tokenizer.bin   # wrong invocation -> FAIL (usage error)
python3 check.py -- python3 -I run.py                           # correct invocation (run was cut off
                                                                # when the session ended; re-run below)
```

Profiling and optimisation (session 2):
```
cp run.py /tmp/p1bench/run_v0.py                                   # baseline snapshot
python3 check.py -- python3 -I run.py          > /tmp/p1bench/check_v0.log    # baseline: PASS, 75.0 s
python3 -I /tmp/p1bench/prof.py /tmp/p1bench run_v0   # cProfile, 6 forwards at pos 0..5
python3 -I /tmp/p1bench/b2.py 768 768                 # matmul variants: unroll, list rows
python3 -I /tmp/p1bench/b3.py 768 768; ... 2048 768   # unroll widths, column unroll
python3 -I /tmp/p1bench/b4.py                         # attention value-sum restructure, pos 500
python3 -I /tmp/p1bench/uniq.py model.bin             # distinct weight values (dedup idea)
python3 -I /tmp/p1bench/listmem.py                    # RSS/time of list-of-rows weights
python3 -I /tmp/p1bench/cmp.py                        # v0 vs v1: bitwise logits + s/token
python3 -I /tmp/p1bench/prof2.py run_v1 1000          # cProfile one forward at pos 1000
python3 check.py -- python3 -I run.py          > /tmp/p1bench/check_v1.log    # optimised: PASS, 57.1 s
python3 -I /tmp/p1bench/run_v{0,1}.py model.bin tokenizer.bin < /tmp/p1bench/work.txt   # protocol diff
```

Second profiling pass (session 3):
```
python3 -I /tmp/p1bench/b5.py            # split kernel cost: iteration vs arithmetic; struct.iter_unpack
python3 -I /tmp/p1bench/b6.py 768 768; ... 768 2048; ... 2048 768   # on-the-fly tolist() rows
python3 -I /tmp/p1bench/ab.py            # real-model A/B: 4-row vs 8-row kernel (3 interleaved reps)
python3 -I /tmp/p1bench/percmd.py        # per-command costs: load, reset, argmax, encode
python3 -I /tmp/p1bench/load.py frombytes|fromfile ; python3 -I /tmp/p1bench/load2.py   # loaders
python3 -I /tmp/p1bench/cmp3.py run_v1 ; python3 -I /tmp/p1bench/cmp3.py run_v3   # init, RSS, logits
python3 check.py -- python3 -I run.py    > /tmp/p1bench/check_v3.log    # PASS, 57.1 s
python3 -I /tmp/p1bench/run_v3.py model.bin tokenizer.bin < work_v3.txt # protocol diff vs v0
```

## What was implemented

- Checkpoint loading exactly as `read_checkpoint`/`memory_map_weights`: 7-int header,
  negative vocab size means an unshared classifier, and the freq_cis block is skipped.
  Weights are kept as one float32 `array('f')`, which is read into memory rather than mmapped.
- `forward()` follows run.c op for op: rmsnorm, then the q/k/v matmuls, RoPE (per-position cos/sin of
  float32 `pos*freq`), and the KV cache write at `pos`. Next comes multi-head attention over 0..pos
  with GQA indexing (`h / kv_mul`), the softmax, the weighted value sum and the wo matmul with
  its residual. Then the FFN: rmsnorm, w1/w3, SwiGLU, w2 and its residual. Finally the last
  rmsnorm and the full classifier at every position (the prompt steps in GENERATE too).
- KV cache: float32 `(layer, seq_len, kv_dim)` arrays, zeroed at the start of every command.
- Tokenizer: same file format, the same `encode` (dummy prefix, UTF-8 codepoint grouping,
  byte fallback +3, greedy highest-score pair merging with first-index tie-break) and the same
  `decode` (BOS leading-space strip, `<0xHH>` raw bytes, as `sscanf` parses them). A dict
  replaces the bsearch over the sorted vocab. Both do an exact string match.
- Protocol: ENCODE, PIECE, GENERATE (greedy argmax with the lowest index winning ties, prompt
  tokens forced, `stop` on token 1, `max`, `context` at seq_len), FIXED, LOGITS (little-endian
  float32, n x vocab), QUIT. stdout is flushed after every line. One thread. No timing output.

## Profiling evidence (session 2)

Machine: 8 cores (only one is used), 8 GiB RAM, Python 3.12.13. The timings are wall-clock.

Baseline (v0) cProfile, 6 forwards at pos 0..5 (19.1 s profiled):

| function | tottime | share |
|---|---|---|
| `matmul` (row slice + `zip` scalar loop) | 18.58 s | 97.3% |
| `f32` (struct round-trips, mostly SwiGLU) + pack/unpack | 0.31 s | 1.6% |
| everything else (forward body, rmsnorm, rf32, exp) | ~0.2 s | ~1% |

Per-token work is ~110M multiply-adds: 85M in the layers and 24.6M in the classifier. So
the only lever that matters is the cost per multiply-add of the interpreted inner loop.

Micro-benchmarks, ns per multiply-add. Every variant produced bit-identical results:

| variant | 768x768 | 2048x768 |
|---|---|---|
| v0: one row per pass, `zip(w[row], x)` | 26.8 | 26.8 |
| 4 rows per pass over x, `zip(x, r0, r1, r2, r3)` (**chosen**) | 20.9 | 20.8 |
| 6 rows per pass | 21.0 | 20.1 |
| 8 rows per pass | 20.8 | – |
| 4 rows x 2 columns (strided slices) | 25.6 | 24.9 |
| 2 rows x 2 columns | 25.8 | 25.6 |
| weights as Python lists, 1 row | 23.2 | – |
| weights as Python lists, 4 rows | 16.6 | – |
| weights as Python lists, 8 rows | 16.1 | – |

Attention value sum at pos 500 (64 x 500): v0 `acc[j] += a*c` loop 36.9 ns/MAC. The
strided-column dot products with 4 columns per pass, same t order, take 24.2 ns/MAC.

After the change (v1), at pos 1000 `dots` is 98% of profiled time. Everything else
(rf32, exp, rmsnorm, forward body) is about 2%.

## Changes made for performance (session 2)

1. **`dots()` kernel, 4 output rows per pass.** One `zip(x, row0, row1, row2, row3)` loop
   updates four accumulators, which amortises the interpreter's per-iteration cost. Each row is
   still summed left to right (j ascending) with the same operations, so results are
   bit-identical to v0. A remainder loop handles row counts that are not a multiple of 4.
   All matmuls and the classifier use it.
2. **Attention through the same kernel.** Scores: rows are the cached keys at stride kv_dim.
   Value sum: column i of the value cache is a strided slice `vc[ko+i :: kv_dim]` over
   t = 0..pos, dotted with the attention weights in ascending t. That is the same summation
   order as C's `xb[i] += a * v[i]` loop, without per-element list indexing.
3. **Batched float32 rounding.** Softmax, rmsnorm and SwiGLU now round each step's vector
   through `array('f')` instead of calling `struct` once per element. The rounding points are
   the same: one per C float store or float expression.
4. **SwiGLU overflow fix.** v0 shortcut `val < -88` to sigmoid = 0. Now `expf(-val)` overflows to
   `inf` exactly as float32 does: `array('f')` casts values above FLT_MAX to inf, and
   `1/(1+inf) = 0`. Values in (-88.72, -88) are now handled exactly as in C. This was not
   hit in the tests below.

Rejected:
- **Weights as Python float lists** (up to ~20% faster). It costs 32 bytes per weight,
  about 3.5 GB for 110M weights on an 8 GiB machine. Measured: maxrss went 0.94 → 2.42 GiB
  even with macOS memory compression, and loading took 4.4 s longer. Swap risk.
- **Deduplicating float objects** to shrink that list layout: 64% of the first 20M weights are
  distinct values, so it would not save enough.
- **Column unrolling:** slower (table above).
- **`math.sumprod` / `sum(map(mul, …))`:** these are C-level dot-product kernels, not
  scalar loops in the P1 sense.
- **Skipping the classifier on forced prompt steps:** that would reduce the work (RULES 1).

## Second profiling pass (session 3)

Starting point: v1, where `dots()` is 98% of forward time. I split the kernel's
cost into parts on 768x768, in ns per multiply-add:

| measurement | ns/MAC |
|---|---|
| v1 kernel (4 rows per pass, array slices) | 20.2 |
| the same loop with an empty body (iteration + float boxing only), array slices | 9.9 |
| the same loop with an empty body, pre-built list rows | 5.7 |
| interleaved weights + `struct.iter_unpack('<4f')` | 23.9 |
| interleaved weights + `struct.iter_unpack('<8f')` | 21.8 |
| on-the-fly `slice.tolist()` per row (768x768 / 768x2048 / 2048x768) | 27.3 / 27.4 / 27.5 |
| on-the-fly `tolist()` of a 4-row block | 29.0 |
| whole-matrix `tolist()` per call | 31.8–33.3 |

About half the cost is the interpreted `v += a*c` arithmetic, which can't be avoided in
scalar Python. The other half is iterating and boxing the float32 elements. Every alternative
way of producing those boxed floats on the fly is slower than iterating `array` slices,
because array iteration reuses freed float objects. Only weights kept permanently as lists
are faster (rejected in session 2 for memory).

Real-model A/B, 6 forwards, best of 3 interleaved repetitions:

| kernel | s/token (pos 0..5) | forward at pos 1000 |
|---|---|---|
| 4 rows per pass (v1, kept) | 2.277 | 2.88 s |
| 8 rows per pass | 2.330 | 3.22 s |

The logits were bit-identical, but the 8-row version is slower, so I rejected it.

Per-command costs (best of 10): `reset()` 3.4 ms (2.6 ms with in-place zeroing), argmax over
32000 logits 1.4 ms, ENCODE of a 15-token sentence 0.2 ms. All are below 0.2% of one forward, so
I left them alone. Model loading was the only other measurable cost:

| loader | load s | peak RSS |
|---|---|---|
| v1 `array.frombytes(f.read())` | 0.14–0.65 | 1.14–1.24 GiB |
| `array.fromfile` | 0.09 | 0.83 GiB |
| pre-sized `array('f',[0.0])*n` + `readinto` (**chosen**) | 0.10–0.20 | 0.47–0.49 GiB |

Full `Transformer.__init__` (v1 vs v3): 0.79 s → 0.11 s, peak RSS 1.25 → 0.53 GiB.

### Change made (session 3)

5. **Loader:** the checkpoint is read straight into a pre-sized float32 array with
   `readinto`. In session 4 this became `readinto` on the array itself, because PyPy rejects
   the memoryview slice used here at first. There is no temporary 438 MB `bytes`
   object. The KV caches are allocated as `array('f',[0.0])*N` instead of from a temporary
   `bytes(4N)`. Compute is unchanged, so logits and protocol output are bit-identical to v0/v1.

No compute-kernel change survived measurement in this pass. Per-token speed is the same as v1.

## Tests and results

| test | v0 (baseline) | v1 (current) |
|---|---|---|
| `python3 check.py -- python3 -I run.py` | PASS, 75.0 s | **PASS, 57.1 s (−24%)** |
| max logit error reported by check (pos 0 / 4 / 12) | 1.14e-5 / 1.53e-5 / 1.62e-5 | identical |
| forward, short context (8 tokens, pos 0..7) | 2.93 s/token | 2.27 s/token (−22%) |
| forward at pos 1000 (alone) | – | 2.89 s |
| own protocol workload, 45 forwards (work.txt) | 136.4 s | 105.5 s (−23%) |
| logits v0 vs v1 (8 positions) and LOGITS files | – | bitwise identical |
| full protocol stdout v0 vs v1 (ENCODE incl. UTF-8, PIECE, FIXED, LOGITS, GENERATE) | – | identical |

Session 3 (v3 = current `run.py`, v1 kernel + new loader):

| test | v1 | v3 (current) |
|---|---|---|
| `python3 check.py -- python3 -I run.py` | PASS, 57.1 s | **PASS, 57.1 s** (same errors 1.14e-5 / 1.53e-5 / 1.62e-5) |
| model init | 0.79 s | 0.11 s |
| peak RSS (init + 8 forwards) | 1.25 GiB | 0.53 GiB |
| forward, pos 0..7 | 2.318 s/token | 2.296 s/token (within noise) |
| own protocol workload (work.txt) | 105.5 s | 105.5 s |
| logits (8 positions), protocol stdout and LOGITS file vs v0 | identical | identical |

The protocol workload (/tmp/p1bench/work.txt) is made from my own prompts, not the checker's
inputs. It contains: 3 ENCODE (ASCII, UTF-8 with byte fallback, empty), 3 PIECE,
FIXED of 10 ids, LOGITS of 5 ids, `GENERATE 24 Once upon a time`, and `GENERATE 1 The cat`.

## Session 4: fixes for two failed external checks

Reported failures:
1. `PIECE 0 3` and `PIECE 1 3` answered `HEX 00`. The reference answers `HEX ` (empty piece).
2. `pypy3 -I run.py` (PyPy 7.3.21, Python 3.11.15) crashed at load: `f.readinto(mv[got:])`
   raised `TypeError: a read-write bytes-like object is required, not 'memoryview'`.

Commands:
```
pypy3 -I /tmp/p1bench/pypyload.py ; python3 -I /tmp/p1bench/pypyload.py   # loader variants
python3 -I run.py model.bin tokenizer.bin < work_cpy.txt  > out_cpy.txt    # protocol script
pypy3   -I run.py model.bin tokenizer.bin < work_pypy.txt > out_pypy.txt   # same commands
pypy3 -I /tmp/p1bench/pk.py ; python3 -I /tmp/p1bench/pk.py               # kernel shapes
python3 check.py -- python3 -I run.py
python3 check.py -- /Users/Shared/pvc-kit/py/pypy-3.11.15-macos-aarch64-none/bin/pypy3 -I run.py
```
(`/tmp/p1bench` had been cleaned between sessions. The scripts above were recreated, and the
work script was rebuilt with the same commands plus extra PIECE cases.)

Fixes:
1. **decode NUL semantics:** `Tokenizer.decode` cuts the piece at the first NUL byte, as C's
   `char*` does. So `<0x00>` decodes to the empty piece and the answer is `HEX ` with nothing
   after it. Other byte pieces are unchanged (`PIECE 1 4` gives `HEX 01`).
2. **Portable loader:** the loader calls `f.readinto(data)` on the pre-sized `array('f')`
   itself, without a sliced or cast memoryview, and falls back to `array.fromfile` on a short
   read. Loader tests, 438 MB checkpoint:

   | loader | CPython 3.12 | PyPy 7.3.21 |
   |---|---|---|
   | `readinto(array)` (**chosen**) | ok, 0.10 s | ok, 0.13 s |
   | `readinto(memoryview(a).cast('B')[0:])` (session 3) | ok, 0.08 s | **TypeError** |
   | `array.fromfile` | ok, 0.09 s | ok, 0.43 s |

3. **PyPy kernel:** with the crash fixed, PyPy ran the protocol script about 2.5x slower than
   CPython (265.8 s vs 107.0 s). The zip-over-array-slices loop tuned for CPython is a bad
   shape for PyPy's JIT. Micro-benchmark at 768x2048, ns per multiply-add:

   | kernel | PyPy | CPython |
   |---|---|---|
   | zip over 4 array slices (`_dots_zip`) | 47.8 | 20.6 |
   | indexed loop, 1 row per pass | 4.6 | 46.9 |
   | indexed loop, 4 rows per pass (`_dots_indexed`) | 3.9 | 39.2 |

   `run.py` keeps both and picks one at import time with `sys.implementation.name`.
   Both are explicit scalar loops with the same per-row left-to-right summation, so the
   results are bit-identical (asserted in the benchmark, and confirmed end to end below).

Results:

| test | result |
|---|---|
| PIECE 0 3 / PIECE 1 3 (CPython and PyPy) | `HEX ` / `HEX ` |
| protocol script stdout, CPython vs PyPy | identical |
| LOGITS file (5 x 32000), CPython vs PyPy | bitwise identical |
| protocol script wall time | CPython 107.0 s, PyPy 265.8 s → **23.4 s** with the indexed kernel |
| `check.py -- python3 -I run.py` | PASS, 58.8 s (errors 1.14e-5 / 1.53e-5 / 1.62e-5) |
| `check.py -- pypy3 -I run.py` | PASS, 10.9 s (same errors) |

The supplied `check.py` passed even with the old `HEX 00` answer, so it doesn't cover that
case. I verified it directly with the PIECE lines above.

## Precision (limitation)

The Python standard library has no float32 scalar arithmetic. All stored tensors are float32:
weights, activations after each op, q/k/v, the KV cache, attention weights and logits.
Scalar steps in rmsnorm, RoPE, softmax and SwiGLU are rounded to float32 in the reference's
order, with `struct` for scalars and `array('f')` for vectors. **The running sums of the dot
products (matmuls, attention scores and the weighted value sum) and of the rmsnorm and softmax
sums are Python floats (binary64), rounded to float32 when stored.** They accumulate left to
right in the same order as the C loops. Rounding every partial sum to float32 would need a
round-trip per multiply-add, about 40x slower. That makes this ~110M-parameter model unusable,
so I did not do it. As a result the logits differ from the C reference by float32-level
rounding (max ~1.6e-5 in the check), not bit-exactly. `math.exp/cos/sin/sqrt/pow` are
double-precision libm calls rounded to float32. `expf` etc. may differ by 1 ulp.

## Other choices and limitations

- "Scalar loops": the inner loops are explicit Python `for` loops with one multiply and one
  add per weight. No `math.sumprod`, `sum(map(...))` kernels or vectorised helpers are used.
- Speed: ~2.3 s per token at short context, ~2.9 s at pos 1000. Long GENERATE or FIXED runs
  take minutes per 100 tokens. Since `dots()` is 98% of the time, further gains within P1
  would need a different weight representation (rejected above for memory) or a newer
  interpreter.
- Peak memory is about 0.53 GiB: the 0.44 GB weight array plus two 38 MB KV caches, which are
  reallocated per command.
- After two profiling passes, the kernel is at its measured floor for array-backed weights in
  CPython 3.12: about 10 ns/MAC of iteration and boxing plus about 10 ns/MAC of arithmetic.
  The only faster layout I measured is weights kept permanently as Python float lists
  (about 16 ns/MAC, roughly 20% faster). It costs about 3.5 GB, which I judged too much
  for this 8 GiB machine.
- PIECE follows C string semantics: decoded pieces end at the first NUL byte, so the
  `<0x00>` byte token (id 3) answers `HEX ` with nothing after it, like the reference. Before
  session 4 it answered `HEX 00`, which failed an external check.
- On PyPy, compute uses a different (indexed) scalar kernel, selected at import time. Both
  kernels give bit-identical results.
- Inputs longer than seq_len: FIXED/LOGITS process only the first seq_len ids. GENERATE reports
  `context`. `GENERATE 0 …` answers `DONE 0 max` without running the model.
