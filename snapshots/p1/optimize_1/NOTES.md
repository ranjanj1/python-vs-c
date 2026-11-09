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

The protocol workload (/tmp/p1bench/work.txt) is made from my own prompts, not the checker's
inputs. It contains: 3 ENCODE (ASCII, UTF-8 with byte fallback, empty), 3 PIECE,
FIXED of 10 ids, LOGITS of 5 ids, `GENERATE 24 Once upon a time`, and `GENERATE 1 The cat`.

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
- Peak memory is about 1 GiB: the 0.44 GB weight array plus two 38 MB KV caches, which are
  reallocated per command.
- PIECE for the `<0x00>` byte token prints `HEX 00` (the decoded byte). C's `safe_printf`
  would treat that piece as an empty string.
- Inputs longer than seq_len: FIXED/LOGITS process only the first seq_len ids. GENERATE reports
  `context`. `GENERATE 0 …` answers `DONE 0 max` without running the model.
