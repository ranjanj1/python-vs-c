# CB port: C + Apple Accelerate BLAS

## Files
- `cb.c`: the implementation (single file). `cb` is the built binary.
- Nothing in the harness, fixtures, rules or `reference/` was changed.

## Build
```
clang -O2 -Wall -Wextra -o cb cb.c -framework Accelerate              # normal build
clang -O2 -Wall -DCB_PROFILE -o cb_prof cb.c -framework Accelerate    # optional profiling build
```
No `-ffast-math` or `-Ofast`. The code starts no threads and sets no thread counts, so Accelerate runs with whatever the harness environment gives it. With `-DCB_PROFILE`, per-section totals go to stderr on QUIT. Without it, the timer macros compile to nothing.

## Model (from the checkpoint header)
dim 768, hidden 2048, 12 layers, 12 heads, 12 kv heads, vocab 32000 (positive, so the classifier shares the embedding weights), seq_len 1024. That is about 110M float32 parameters, ~438 MB read per forward pass.

## What matches the reference
- **Same weight values, float32 throughout.** Shared or unshared classifier is chosen from the sign of vocab_size. The tensor offsets in the file are exactly the reference's `memory_map_weights`, including the skipped legacy freq_cis block.
- **Matmuls use the same shapes.** Each `matmul(W(d,n) @ x)` becomes one `cblas_sgemv(RowMajor, NoTrans, d, n, ...)`: q/k/v/o, w1/w3/w2 and the classifier, one token at a time (batch 1). There is no prompt batching into GEMM and no fusing of q/k/v or w1/w3 into larger matrices.
- **Attention uses BLAS matvecs over the reference KV cache layout** `(layer, seq_len, kv_dim)`. Per head, scores are `K[0..pos] @ q` with sgemv (lda = kv_dim), then the reference's `/ sqrtf(head_size)` and softmax. The weighted value sum is `V[0..pos]^T @ att` with transposed sgemv.
- **Scalar code is copied from the reference:** rmsnorm, softmax, RoPE formula, SwiGLU, residuals and argmax (strict `>`, so the lowest index wins ties), with the same `expf/powf/cosf/sinf`.
- **The tokenizer is the reference's.** BPE encode is the reference algorithm, including its UTF-8 handling, byte fallback and dummy prefix. `PIECE` uses the reference `decode`: it strips the leading space after BOS and maps `<0xXX>` to the raw byte. It prints the bytes up to the C string terminator, so `<0x00>` gives `HEX ` with nothing after it.
- **GENERATE follows the reference loop.** Prompt tokens are forced, and each command starts at pos 0. Greedy choice. `stop` if the chosen token is 1, `max` once max_new tokens are printed, `context` when the position reaches seq_len. If max_new and context are reached on the same token, it reports `max`.
- **The KV cache is clean for every command:** before each GENERATE/FIXED/LOGITS, the rows the previous command wrote are zeroed.

## Changes that don't change results
1. **Weights loaded with `pread` into one page-aligned buffer, laid out in access order** (second session). The layout is the embedding/classifier table, then per layer `rms_att, rms_ffn, wq, wk, wv, wo, w1, w3, w2`, then rms_final. In the file, the weights start at byte offset 28 and each tensor type is stored for all layers together, so a forward pass jumped between 7 separate regions. Now one forward streams one buffer from start to end (plus the classifier). Values are unchanged. Logits over 1024 positions are bit-identical to the first-session aligned-copy build (`cmp.py`: max |diff| 0).
2. **RoPE angles computed once per forward pass.** cos/sin are computed with the exact reference formula once per forward instead of once per layer. They don't depend on the layer, so the values are bit-identical.
3. **Vocab sorted at load.** The sorted vocab for encode is built at load instead of lazily on the first ENCODE.

## Profiling (second session)
Machine: Apple M1 (4 performance + 4 efficiency cores), 8 GB. It was **heavily shared** during this work: load average 3.3–5.4, with one spike to 15.2. That makes the timings noisy (see Limitations). Benchmarks ran with `VECLIB_MAXIMUM_THREADS=1` unless noted. The only development workloads used are the supplied ones: the 3 fixture prompts (GENERATE 64), FIXED/LOGITS on their fixture `seq`s (69–82 tokens), and a 1024-position FIXED built by repeating a fixture `seq`, to cover long context.

### Where the time goes
Measured with the `-DCB_PROFILE` build: 2061 forwards of the fixture workload, after the layout change.

| section | ms/forward | share |
|---|---|---|
| w1/w3 sgemv (2 × 2048×768) | 5.12 | 32.3% |
| classifier sgemv (32000×768) | 3.48 | 22.0% |
| w2 sgemv (768×2048) | 3.01 | 19.0% |
| wq/wk/wv sgemv (3 × 768×768) | 2.90 | 18.3% |
| wo sgemv (768×768) | 0.97 | 6.1% |
| attention (scores, softmax, values) | 0.22 | 1.4% |
| swiglu (expf) | 0.11 | 0.7% |
| rmsnorm, rope, residual, embedding | 0.06 | 0.4% |

- **About 98% of the time is in the weight `sgemv` calls, and each takes time in proportion to its weight bytes** (25–29 GB/s for every matrix). So the program is memory-bandwidth bound in Accelerate's sgemv, and the scalar code is not worth optimising.
- **At 1024 positions** (1024 forwards, FIXED), attention grows to 2.5 ms/forward (15%). The weight matmuls are unchanged.

### BLAS floor microbenchmark (`/tmp/cbtest/mb.c`)
This benchmark runs one token's worth of distinct matrices with the same shapes (~440 MB, too large to stay in cache), best of 15 runs:
- `sgemv RowMajor NoTrans`: 11.9–13.5 ms/token (≈33–37 GB/s)
- `sgemv ColMajor Trans` (same storage): same within noise
- `sgemv` on transposed storage: about 5–10% slower
- `sgemm` with N=1: same or slightly faster in isolation, but not faster inside the program (interleaved A/B: 14.83 vs 13.29 ms median)

Default threading was no faster than `VECLIB_MAXIMUM_THREADS=1`. So about 12–13 ms per token is the floor for this BLAS on this machine, and the program is at about 13–14.5 ms per forward in quiet periods.

### A/B results
Both harnesses are interleaved: they alternate between the builds and reverse the order every round.
- `ab.py`: FIXED over all 3 fixture seqs, median ms/forward.
- `ablong.py`: one 1024-position FIXED.

| comparison | result |
|---|---|
| original mmap → aligned copy → access-ordered (`ab.py`, 8 rounds, load 4.1) | 16.48 → 15.26 → **13.29** |
| original mmap vs access-ordered (8 rounds, load 3.4) | 14.22 vs **12.92** |
| original mmap vs final (8 rounds, load 3.3) | 14.40 vs 14.64 (tie, within noise) |
| final vs original mmap, reversed launch order (6 rounds, load rising to 15) | **16.78** vs 20.01 |
| sequential `work.py` runs, 2 reps each | original 61.2 vs final **70.2** fwd/s; GENERATE 3.86 vs **3.32** s per rep; first token 89 vs **69** ms |
| load until READY | original 0.03 s (lazy mmap, page faults paid later) / aligned memcpy ~0.9–1.5 s / final `pread` **0.07–0.43 s** |

### Tried and rejected
KV-cache layouts. Each was tested with `ab.py` / `ablong.py`; numbers are median ms/forward on the short workloads.

| layout | short workloads | 1024 positions | max \|Δlogit\| at 1024 positions |
|---|---|---|---|
| reference layout (kept) | 12.85–13.15 | 15.13–16.93 | — |
| head-major K/V `[layer][kv_head][pos][hs]` | 14.11 | 17.13 | 0 |
| head-major K + V transposed per head (NoTrans value sgemv) | 14.13–14.28 | 15.22–16.54 | 9e-6 |
| same as above, with padded row stride (seq_len+16) | 13.75 | 16.54 | — |

- **Why they were tempting:** the in-process profile showed attention at 1024 positions dropping from 2.5 to 1.8 ms/forward with head-major K/V. The Trans value sgemv took 1.13 s, against 0.46 s for the scores over the same data.
- **Why they were rejected:** every variant was 0.6–1.4 ms slower per forward on the short development workloads, and none gave a clear win at 1024 positions. The likely cause is scattered cache lines and 4 KB set aliasing at small `pos`. **I reverted to the reference layout.**

Also not adopted:
- Fusing wq/wk/wv or w1/w3 into one larger sgemv: changes the operation shapes.
- Batching prompt tokens into sgemm: violates batch size 1 and same shapes.
- vForce `vvexpf` for SwiGLU or softmax: not BLAS, different rounding, and the gain is ≤0.7%.
- `mlock`ing the weights: they could be compressed while idle on this 8 GB machine, but pinning 438 MB on a shared machine wasn't justified.

## Tests run (final build)
```
python3 check.py -- ./cb                                   # PASS
python3 check.py --tokens 64 --prompts 3 -- ./cb           # PASS, max logit error 1.05e-5 .. 2.86e-5 (tol 5e-4)
python3 /tmp/cbtest/extra.py $PWD                          # PIECE vs all fixture pieces, FIXED argmax vs fixture ids,
                                                           # byte pieces, GENERATE 0, empty prompt, stop on token 1,
                                                           # QUIT exit 0 -> all OK
python3 /tmp/cbtest/cmp.py $PWD /tmp/cbtest/v_cur ./cb     # 1024-position LOGITS: bit-identical to the pre-experiment build
```
- **Context stop:** tested in the first session (691-token prompt → `DONE 334 context`). The final build uses the same generation loop.
- **Unshared-classifier models:** the loader reads `wcls` from the reference offset when vocab_size < 0. This path is untested because only a shared-weights checkpoint is supplied.

### Commands, first session
```
cat RULES.md INTERFACE.md check.py reference/run.c; inspect fixtures.json + model header
clang -O2 -Wall -Wextra -o cb cb.c -framework Accelerate; python3 check.py ...
python3 /tmp/cbtest/bench.py ...            # mmap vs aligned-copy weights
```

### Commands, second session
```
sysctl machdep.cpu.brand_string hw.ncpu ...; vm_stat; uptime
python3 /tmp/cbtest/work.py $PWD <exe> N                      # fixture workload driver
clang -O2 -DCB_PROFILE -o /tmp/cbtest/cbp cb.c -framework Accelerate; work.py with cbp
clang -O2 -o /tmp/cbtest/mb /tmp/cbtest/mb.c -framework Accelerate; ./mb   # BLAS floor
python3 /tmp/cbtest/ab.py $PWD R <exe...>; python3 /tmp/cbtest/ablong.py $PWD R <exe...>
python3 /tmp/cbtest/cmp.py $PWD <exeA> <exeB>
```
The `/tmp/cbtest` scripts and variant builds are throwaway drivers. They are not part of the deliverable.

## Limitations / notes
- **The measured gain is uncertain.** It ranges from a tie to about 20% in the interleaved A/B runs. My launch-order bias check (the same binary against itself) was never run, so a harness bias can't be ruled out.
- **The program is memory-bandwidth bound in Accelerate sgemv** (~98% of the time). It is within about 10% of the measured BLAS floor for these shapes. Further gains would need fewer bytes per token (quantization or reduced work) or different operation shapes, and the rules forbid both.
- **Results are not bit-exact with the reference.** Accelerate's sgemv sums in a different order from the reference's serial loop, so logits differ slightly (≤3e-5 on the public fixtures). An argmax near-tie could in principle flip on other inputs.
- **Accelerate threading is not controlled by the program.** It may use multiple threads unless the harness limits it (e.g. `VECLIB_MAXIMUM_THREADS`). Default threading was not measurably faster here.
- **Startup costs more memory.** READY now waits for a full 438 MB `pread`, 0.07–0.43 s measured. In exchange, there are no page-fault stalls in the first commands. Peak memory is about 438 MB of weights plus a 75 MB KV cache.
- **Inputs the interface doesn't define:**
  - FIXED/LOGITS with more than seq_len ids process only the first seq_len.
  - Out-of-range ids, or an unwritable LOGITS path, answer `ERR ...`.
  - Unknown commands answer `ERR unknown command`.
  - Blank lines are ignored.
