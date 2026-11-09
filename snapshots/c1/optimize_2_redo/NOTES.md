# C1 port — notes

Technique set **C1**: C, compiler optimisation flags, CPU intrinsics (ARM NEON). No external math
library, no BLAS, no threads, no fast-math.

## Files
| File | Purpose |
|---|---|
| `run_c1.c` | The implementation (single file). |
| `build.sh` | `./build.sh` builds `./run_c1` (the submission). `./build.sh profile` builds `./run_c1_prof`: the same code with section timers (`-DPROFILE`), which reports to stderr at QUIT. The timers are compiled out of the normal build. |
| `tests/ref_encode.c` | Differential helper: `#include`s the unmodified `reference/run.c` (with `-DTESTING`) and prints its `encode()`/`decode()` output. |
| `tests/ref_logits.c` | Differential helper: the unmodified reference `forward()`; writes all logits for an id sequence. |
| `tests/difftest.py` | Compares ENCODE (2009 strings, including random ASCII/UTF-8/invalid bytes and a 3000-char string) and PIECE (all 3×32000 pairs with `prev` ∈ {0,1,2}) against the reference. |
| `tests/longctx.py` | Compares LOGITS against the reference over a full 1024-position sequence, checks FIXED argmaxes, and covers GENERATE `stop` and `GENERATE 0`. |
| `tests/context.py` | Exercises the `context` stop with prompts of 981, 1006 and 1101 tokens. |
| `tests/paths.py` | Exercises paths the supplied model doesn't use: an unshared classifier (synthetic checkpoint with negative vocab_size) and the unpacked row-major fallback (`-DPR=3` build). |
| `tests/membw.c`, `tests/membw2.c` | Memory-bandwidth and matmul-kernel microbenchmarks used as profiling evidence. |
| `tests/bench.py`, `tests/loadtime.py` | Rough local timing only. The harness does the real timing. |

Nothing in the supplied bundle (harness, fixtures, reference, rules) was modified. No hidden inputs,
external implementations or other ports were used. The only workloads were the supplied
`check.py`/`fixtures.json`, plus synthetic id sequences and prompts generated in `tests/`.

---

## Pass 1: initial port (summary)
- **Same work as the reference:**
  - The same checkpoint layout, including shared and unshared classifier through the negative
    vocab_size, and the skipped legacy RoPE tables.
  - float32 weights and activations, and the same operations per layer.
  - Greedy argmax with the lowest index winning ties.
  - Batch size 1 and **one token per forward pass**, positions 0..n-1. Prompt tokens are forwarded one
    at a time like the reference; there is deliberately no multi-position prefill batching.
- **No libm:**
  - `sqrt` uses the `fsqrt` instruction.
  - `exp` is my own vectorised float32 implementation (Cody–Waite reduction plus a degree-7 polynomial,
    with overflow, underflow and NaN handling).
  - RoPE cos/sin come from tables built at load. The angle is formed in float32 exactly as in the
    reference (`freq = 1/pow(10000, hd/hs)`, `val = pos*freq`), then evaluated with my own
    double-precision exp and sin/cos. Double is used only for that load-time table.
  - `otool -L` / `nm -u` show only libSystem and no math symbols.
- **Tokenizer:** the reference algorithm, with hash lookups instead of bsearch and incremental pair
  rescoring. Output is identical, verified by `tests/difftest.py`.
- **Initial kernel:** a row-major NEON matvec, 2 rows per pass with 64-byte loads. The 2-row choice came
  from benchmarking 1/2/4/8-row variants: 2 rows gave 8.75 ms/forward against 11.8 ms for the first
  4-row version.

## Pass 2: profiling and optimisation

### Profiling method
- **Section timers:** `./build.sh profile` adds `mach_absolute_time` timers around every stage of
  `forward()`, plus argmax, encode and the LOGITS file write.
- **Dev workload:** `python3 check.py --tokens 64 --prompts 3 -- ./run_c1_prof`. This is all supplied
  fixtures: 9 ENCODEs, 3× GENERATE 64, and 3 LOGITS prefixes, 289 forwards in total.
- **Long context:** a 1024-position `FIXED` run with random ids (`seed 7`), to see how cost grows with position.
- **Kernel ceiling:** microbenchmarks over model-shaped matrices (`tests/membw*.c`) to find the
  single-core read roofline.

### Evidence: baseline (pass-1 code), dev workload, 289 forwards
| section | ms/forward | share | effective BW |
|---|---|---|---|
| matmul q,k,v (85 MB) | 1.626 | 18.3% | 52 GB/s |
| matmul wo (28 MB) | 0.537 | 6.0% | 53 GB/s |
| matmul w1+w3 (151 MB) | 2.835 | 31.9% | 53 GB/s |
| matmul w2 (75 MB) | 1.744 | 19.6% | **43 GB/s** |
| matmul classifier (98 MB) | 1.839 | 20.7% | 53 GB/s |
| attention | 0.221 | 2.5% | |
| everything else | <0.1 | <1% | |
| **total** | **8.89** (2569 ms) | | |

- Matmuls were 96.5% of the time, so the work is weight-streaming bound (438 MB per token).
- At 1024 positions, attention grew to **3.89 ms/forward on average (34.7%)**. That is about 6× slower
  than the bandwidth needed to read the KV cache.

### Evidence: microbenchmarks (`tests/membw.c`, `tests/membw2.c`, M1, GB/s, best of 7)
| kernel | 768×768 ×48 | w1-like 2048×768 ×12 | w2-like 768×2048 ×12 |
|---|---|---|---|
| pure sequential read+add | 51.2 (*) | 62.6 | 57.2 |
| row-major, 2 rows (pass-1 kernel) | 54.6 | 58.1 | 51.3 |
| packed panels, 1 row, 4 acc | 50.4 | 51.5 | 41.3 |
| packed panels, 2 rows, 4 acc | 61.2 | 61.6 | 47.5 |
| packed panels, 4 rows, 4 acc | 55.4–62.9 | 61.4–62.0 | 58.8–62.7 |
| packed panels, 8 rows | 60.7 | 55.9 | 46.5 |
| packed 2 rows × 8 acc | 61.0–61.1 | 61.8–61.9 | 49.5–62.4 |

(*) The read+add loop is itself add-latency limited on this shape, so it is not a true ceiling.

- A single stream with only 4 accumulators stalls at ~51 GB/s: it is FMA-latency bound.
- Several rows interleaved into one sequential stream reach about 62 GB/s, which is the practical
  single-core ceiling here.
- Results vary by a few GB/s between runs, and ranges show two rounds. The w2 shape in particular was
  slow in the first round and ~62 GB/s in the second, so I treat ±3 GB/s as noise.

### Changes made
1. **Packed, sequential weight stream.**
   - At load, every matmul weight is repacked into panels of 4 rows. Within each panel, 16-float blocks
     of the 4 rows are interleaved.
   - The panels are laid out back-to-back in exact consumption order: per layer wq, wk, wv, wo, w1, w3,
     w2, then the classifier.
   - A forward pass therefore reads all weights as one contiguous stream. The kernel keeps 16 independent
     FMA accumulators (4 rows × 4).
   - The values are the same float32 weights, and each output row is still one dot product over the same
     inputs; only memory order changed.
   - Shapes that can't be packed (`n % 16`, `d % 4`) keep the row-major path.
   - When the classifier is shared, the token embedding row is gathered back out of the packed
     classifier, so there is no second copy.
2. **Load via mmap plus a single packing copy.** This replaces `read()` into a buffer followed by an
   alignment memcpy. Time to READY fell from ~260 ms to ~95–120 ms (median), and resident memory is
   unchanged (one copy of the weights).
3. **Head-major KV cache, `[layer][kv_head][pos][head_size]`.**
   - Each head's keys and values for positions 0..pos are contiguous, instead of 256-byte slices with a
     3 KB stride.
   - The cache stores the same k and v (after RoPE for k), written once per position and read for
     positions 0..pos, exactly as in the reference. Only the index order changed.
4. **Blocked attention kernels.**
   - Scores are computed for 4 positions at once with 16 independent accumulators and one vector divide
     by `sqrt(head_size)`. The divide is kept as a divide, like the reference.
   - The weighted V sum keeps a 64-float output chunk in 16 registers.
   - The softmax max uses a vector `vmaxq`, which is exact.
5. **Vectorised argmax.**
   - A vector pass finds the max value, then the code takes the first index equal to it. That equals the
     reference's strict-`>` scan, including ±0.
   - If any NaN is present, it falls back to the reference scan.
6. **Tried and rejected** (measured in the real binary with `tests/bench.py`; run-to-run noise is
   about ±0.4 ms/forward):
   - 2-row panels: equal to 4-row within noise.
   - 8-row panels: 0.15–0.8 ms/forward slower.
   - Software prefetch at 256/1024 bytes ahead: no consistent gain. At 4096 bytes: about +1.5 ms/forward.

### Evidence: after (same profiling runs)
Dev workload (`check.py --tokens 64 --prompts 3`, 289 forwards):

| section | before ms/fwd | after ms/fwd | after BW |
|---|---|---|---|
| matmul q,k,v | 1.626 | 1.411 | 60 GB/s |
| matmul wo | 0.537 | 0.467 | 61 GB/s |
| matmul w1+w3 | 2.835 | 2.494 | 61 GB/s |
| matmul w2 | 1.744 | 1.244 | 61 GB/s |
| matmul classifier | 1.839 | 1.621 | 61 GB/s |
| attention | 0.221 | 0.136 | |
| argmax | 0.027 | 0.004 | |
| **total** | **8.89 (2569 ms)** | **7.47 (2160 ms)** | **−16%** |

1024-position FIXED run:

| | before | after |
|---|---|---|
| attention ms/forward (average) | 3.885 | 0.822 |
| total | 11455 ms | 8356 ms (−27%) |

Other timings (wall clock, noisy):

| measurement | before | after |
|---|---|---|
| LOGITS 1024 (`tests/longctx.py`) | 12.5 s | 8.6 s |
| FIXED 512 (`tests/bench.py`) | 5.06 s | 4.09 s |
| READY | ~260 ms | ~100–120 ms |
| full `check.py --tokens 64 --prompts 3`, including Python | | 2.4–2.9 s |

All matmuls now run at ~60–61 GB/s, essentially the ~62 GB/s single-core roofline measured above.
What remains is DRAM bandwidth.

### Tests run after the changes (all on the final code)
```
./build.sh && ./build.sh profile
python3 check.py -- ./run_c1                          # PASS (max logit err 1.7e-05)
python3 check.py --tokens 64 --prompts 3 -- ./run_c1  # PASS (max logit err 2.7e-05, tol 5e-4)
python3 -I tests/difftest.py "$PWD"       # 98009 answers compared, 0 mismatches
python3 -I tests/longctx.py "$PWD" 1024   # worst logit err 5.05e-05 over 1024 positions; argmax 1024/1024
python3 -I tests/context.py "$PWD"        # DONE 44 / 19 / 0 context, as expected
mkdir -p /tmp/c1paths && python3 -I tests/paths.py "$PWD" /tmp/c1paths
                                          # unshared checkpoint: diff 0; PR=3 fallback build: diff 0
cc -O3 -mcpu=apple-m1 -o /tmp/membw tests/membw.c && /tmp/membw
cc -O3 -mcpu=apple-m1 -o /tmp/membw2 tests/membw2.c && /tmp/membw2
python3 -I tests/bench.py "$PWD" ./run_c1 ; python3 -I tests/loadtime.py "$PWD" ./run_c1
```

---

## Limitations / judgement calls
- **Not bit-identical to the reference.**
  - The reduction order differs (vector accumulators), FMA is used (clang's default
    `-ffp-contract=on`, not fast-math), and exp/sin/cos are my own implementations.
  - The max-abs logit difference is ≤ 5.1e-5 over 1024 positions. All argmaxes agreed in the tests
    here, but a near-tie in logits could in principle flip a greedy choice.
- **Remaining headroom is small.** Decoding is bound by single-core DRAM bandwidth (~60 of ~62 GB/s).
  - Larger gains would need fewer bytes per token (quantization, skipping work) or more cores. Both are
    forbidden.
  - Multi-position prefill batching (a GEMM over prompt positions) would speed up FIXED, LOGITS and
    prompt processing a lot. I did not do it because the rules say batch size 1 and the same per-step work.
- **Memory layouts are reordered:** the packed weight panels and the head-major KV cache. They hold the
  same values used in the same computations; I consider this within "same work, same KV-cache
  behaviour", but it is a judgement call.
- **Startup cost:** the packing adds a ~0.1 s one-time copy at startup, before READY.
- **Platform:** the code is arm64/NEON only, tuned on an Apple M1. Panel height and kernels were picked
  by measurement on this machine.
- **Edge inputs the interface does not define** are handled defensively but arbitrarily:
  - Unknown commands reply `ERROR ...`.
  - Out-of-range ids in FIXED/LOGITS are replaced by 0, and an out-of-range PIECE id prints `HEX `.
  - FIXED/LOGITS sequences longer than `seq_len` are truncated.
  - `GENERATE 0` answers `DONE 0 max` without running the model.
  - A decoded 0x00 byte prints as empty (C-string semantics, as in the reference).
- **Hidden gate:** `check.py` passing is necessary but not sufficient. The final gate uses other inputs,
  which I have not seen or used.
