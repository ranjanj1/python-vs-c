# P3 — NumPy float32 port of reference/run.c

Program: `python3 run.py <checkpoint.bin> <tokenizer.bin>` (also executable as `./run.py`).
Profiling scripts live in `prof/`. Frozen snapshots used as baselines: `prof/run_v1.py` (round 1),
`prof/run_v2.py` (end of round 2), `prof/run_v3b.py` (round 3 before QKV/W13 fusion).

## Commands run
### Round 1 (initial port)
```
cat RULES.md INTERFACE.md check.py; read reference/run.c
python3 -c ...            # inspected fixtures.json, model header (768,2048,12,12,12,32000,1024), tokenizer.bin
                          #   (27 max token length, no NUL bytes, no duplicate pieces, 256 <0xNN> byte pieces)
chmod +x run.py
python3 check.py -- python3 run.py                          # PASS (max logit error ~1.9e-5)
python3 check.py --tokens 64 --prompts 3 -- python3 run.py  # PASS (max logit error <= 2.5e-5, tol 5e-4)
printf 'PIECE ...\nFIXED ...\nGENERATE 0 hi\nGENERATE 5 \nENCODE\nQUIT\n' | python3 run.py model.bin tokenizer.bin
printf 'GENERATE 2000 Once upon a time\nQUIT\n' | python3 run.py ...   # DONE 238 stop
python3 - (1021-token prompt, GENERATE 50)                  # DONE 4 context (= 1024 - 1021 + 1)
```
### Round 2 (profiling and optimisation)
```
python3 prof/bench.py <dir> <module>  # ms/token on the 3 fixture sequences + 1024 positions, in-process
python3 prof/parts.py .               # v1: matmul-only time vs full forward, cProfile
python3 prof/variants.py              # gemv layouts: dot(W,x), matmul, dot(x,W^T), fused shapes
python3 prof/bw.py                    # BLAS kernels vs raw read bandwidth; vecdot / einsum / matvec paths
python3 prof/shapes.py                # gemv vs vecdot vs transposed einsum at every model shape, + error vs float64
python3 prof/attn.py                  # attention variants at T=80 and T=1024
python3 prof/parts2.py .              # new: matmul-only vs forward, dtype audit (all float32 / complex64)
python3 prof/tok.py .                 # tokenizer encode time vs prompt length
python3 prof/tokeq.py                 # new encoder vs original run.c-style encoder on 3009 inputs
python3 prof/fuse.py                  # fused QKV / W1W3 matvec vs separate
python3 prof/e2e.py python3 run.py    # end-to-end protocol timing (startup, GENERATE 64 x3, LOGITS x3, long prompt)
python3 check.py -- python3 run.py                          # PASS
python3 check.py --tokens 64 --prompts 3 -- python3 run.py  # PASS (max logit error <= 2.7e-5)
printf '<edge commands>' | python3 run.py model.bin tokenizer.bin    # same answers as round 1
```
### Round 3 (profiling and optimisation)
```
python3 prof/bench.py . run; python3 prof/parts2.py .   # re-profile round-2 state
python3 prof/over.py . run            # min-of-N: matvecs vs forward -> ~1.1 ms non-matvec overhead
python3 prof/sections.py              # per-section timing of the forward pass (instrumented copy)
python3 prof/attn2.py                 # attention layouts, warm cache
python3 prof/attn_insitu.py           # attention variants inside the real forward (cold KV cache)
python3 prof/attn_split.py            # cold attention split: scores / softmax / weighted sum; transposed V
python3 prof/vstride.py               # V layouts and row padding vs context length
python3 prof/dotlen.py                # vecdot and Accelerate gemv throughput vs row length
python3 prof/blocked.py               # column-blocked gemv at every model shape (+ error vs float64)
python3 prof/blocked2.py              # strided-view blocks vs contiguous blocks; reorganisation cost
python3 prof/load.py; prof/load1.py   # loader strategies, in-process and in fresh processes
python3 prof/ready.py python3 run.py  # time from process start to READY (min/median of 7)
python3 prof/wsum_sweep.py            # weighted sum: matmul / einsum on (t,d) vs vecdot on (d,t), T=100..1000
python3 prof/long.py <dir> <module>   # min-of-N ms/token at positions 80/300/600/999, old vs new interleaved
python3 prof/fuse2.py                 # fused vs separate blocked matvecs
python3 prof/blocksize.py 64|128|192|256|384   # in-situ block-size sweep
python3 prof/over2.py                 # final: blocked matvecs vs forward
python3 prof/verify.py                # run.py vs prof/run_v1.py: real model 1024 positions + 3 synthetic checkpoints
python3 prof/e2e.py python3 prof/run_v2.py / run.py   # end-to-end, interleaved
python3 check.py -- python3 run.py                          # PASS
python3 check.py --tokens 64 --prompts 3 -- python3 run.py  # PASS (max logit error <= 2.7e-5)
edge commands through prof/run_v1.py and run.py, diffed      # identical output
printf 'GENERATE 2000 Once upon a time\nQUIT\n' | python3 run.py ...   # DONE 238 stop (unchanged)
```

## Profiling evidence (development workloads = the fixture prompts)
Timings are on a shared machine; during round 2 the machine got noticeably noisier, so I compare
best-of runs and interleaved old/new runs.

v1 forward, prompt 1 at position 80: **13.5 ms**, of which the 85 matrix-vector products took **12.4 ms**
(layers 9.65 ms + classifier 2.78 ms). That is ~35 GB/s on 438 MB of weights per token, and everything
else (attention, norms, RoPE, Python) took ~1.1 ms. The work is memory-bandwidth bound.

Matvec kernel survey (single thread, float32, GB/s of weights streamed):

| shape (d x n) | `np.dot` (Accelerate sgemv) | `np.vecdot` (row dots) | `einsum` on W^T |
|---|---|---|---|
| 768 x 768 | 34 | 43 | 44 |
| 2048 x 768 | 36 | 43 | 46 |
| 768 x 2048 | 31 | 47 | 46 |
| 32000 x 768 | 34 | 43 | 34 |

Plain reads (`ndarray.max`) reach ~56 GB/s, so sgemv was leaving bandwidth unused. sgemm with M=1,
`np.matmul`, `np.matvec` and the transposed layouts all stayed at ~35 GB/s. Max error vs float64
was 1.3e-5 for gemv and 1.5e-5 for vecdot (same order), but 7e-5 to 2.3e-4 for transposed einsum, so I chose vecdot.

Tokenizer: the run.c merge loop is O(n²) in Python: **770 ms** to encode a 1021-token prompt
(about 70 forward passes) and 458 ms for 722 tokens.

Attention at T=1024: ~120 µs/layer (~50 GB/s on the K/V slices), already near memory speed.

## Changes in round 2
1. **Matvecs via `np.vecdot`** (row-wise float32 dot products, `out[i] = sum_j W[i,j] x[j]`, the same
   operation and shape as `matmul()` in run.c; separate Q, K, V, O, W1, W3, W2 and classifier calls as before).
   Matmul time per token went from 12.4 to 10.4 ms.
2. **K and V are written straight into the KV-cache slot** (`out=` a (kv_head, head_size) view of position `pos`),
   and RoPE is applied in place there. This drops the intermediate copies.
3. **RoPE as a complex64 multiply** with a precomputed (seq_len, head_size/2) table built from the same
   float32 formula. `(v0 + i v1)(c + i s)` gives `v0*c - v1*s, v0*s + v1*c`, the reference expressions.
4. **Fewer temporaries / Python calls**: rmsnorm inlined (`w * (ss * x)` order kept), attention output
   written with `out=` into `xb`, per-layer weight views prebuilt.
5. **Tokenizer merge with a lazy-deletion heap** keyed by (-score, original left position). That pair
   is exactly the one run.c's scan picks (highest score, leftmost on ties, strict `>` above float32(-1e10)).
   On 3009 inputs (fixtures, random ASCII/Unicode/tabs, repeated vocab words, random raw bytes) the result is identical
   to the original. Encode time for 1021 tokens dropped from 770 ms to **4.8 ms**.

### Results
- Forward, fixture sequences (best runs): v1 **13.6–14.9 ms/token**, new **10.9–11.1 ms/token** (~20–25 % faster).
  Interleaved runs on the noisy machine: v1 14.1–17.8, new 11.0–16.5 ms/token.
  Matmul share is now 10.4 of 10.9 ms; non-matmul overhead fell from ~1.1 ms to ~0.5 ms.
- End-to-end protocol (`prof/e2e.py`, best of 2): LOGITS on 3 fixture sequences 3.17 → 2.56 s; GENERATE 64 on
  3 prompts 3.10 → 2.81 s; 1021-token prompt to context stop 16.3 → 13.0 s; total 22.9 → 18.7 s.
- Correctness: check.py passes in quick and full modes, with max logit error ≤ 2.7e-5 (tolerance 5e-4). All 64
  generated ids match for all three prompts; the edge-case answers are unchanged.

### Tried and rejected
- Fused QKV and W1|W3 matvecs: only 0.3 ms/token gain (5.92 → 5.62 ms on those matrices). Because each
  weight kind is stored contiguously across layers, fusing needs ~270 MB of copies at load and loses the
  direct-to-cache K/V write. It is a net loss on short workloads.
- Transposed weights with `einsum('j,ji->i')`: similar speed, 5–7× larger rounding error and a 438 MB transpose at load.
- Other attention formulations (`vecdot` for scores) were slower at long context.
- **Not done on purpose:** batched prompt prefill (one GEMM over all forced/FIXED tokens). It would read the
  weights once per prompt instead of once per token and would be much faster. But it changes the per-position,
  batch-1 work and KV-cache filling order that RULES.md asks to keep, so every position still does its own
  matrix-vector forward pass.

## Round 3: profiling evidence
Start (round-2 code), min-of-N in-process timings at position 80: forward 11.15 ms, weight matvecs 10.0 ms,
so ~1.1 ms was something else. Per-section split of one forward (`prof/sections.py`): w1+w3 3.46, classifier 2.19,
qkv 2.02, w2 1.61, wo 0.66, **attention 0.58**, SwiGLU 0.13, RoPE 0.07, norms+residuals 0.13 ms.

**Attention measured in place is much slower than in isolation.** The weight stream evicts the KV cache
between layers. At position 999, attention cost 3.2 ms/token in place versus ~1.4 ms warm. Split with a
cold cache (12 layers, `prof/attn_split.py`): at T=1000, scores (`q·K^T` matmul) 0.73 ms, softmax ~0.5 ms,
and the **weighted sum `att @ V` 2.1 ms** (~17 GB/s, a transposed gemv with only 64 columns).
Weighted-sum sweep over T (`prof/wsum_sweep.py`, ms for 12 layers):

| T | matmul on V(t,d) | einsum on V(t,d) | vecdot on V(d,t) |
|---|---|---|---|
| 100 | 0.33 | **0.29** | 0.58 |
| 300 | 0.69 | **0.53** | 1.35 |
| 600 | 1.23 | **0.87** | 1.44 |
| 800 | 1.62 | 1.12 | **0.95** |
| 1000 | 2.03 | 1.30 | **0.83** |

On V(d,t), each head reads 64 rows of T floats spaced 4 KB apart. Below ~750 positions those short
rows stream badly (padding the stride did not help); near full length they become contiguous.

**Accelerate's gemv speed depends on row length** (`prof/dotlen.py`, 48 MB matrices): gemv reaches 58–60 GB/s
for rows of 128–256 floats but only 27–39 GB/s for 300–4096. `vecdot` gets 40–54 GB/s. So the round-2 vecdot
matvecs (~45 GB/s) could go faster if the matrices were split into short-row column blocks
(`prof/blocked.py`, GB/s):

| shape (d x n) | vecdot | batched gemv, b=128 | b=256 | b=384 |
|---|---|---|---|---|
| 768 x 768 | 43 | 54 | 53 | 42 |
| 2048 x 768 | 44 | 57 | 58 | 44 |
| 768 x 2048 | 47 | 56 | 56 | – |
| 32000 x 768 | 44 | 58 | 58 | 46 |

Max error vs float64 is unchanged (0.9–1.9e-5). Blocks must be contiguous copies: a strided (no-copy) view
gave only 34 GB/s (`prof/blocked2.py`).

## Changes in round 3
1. **Column-blocked weight matvecs.** Every matrix W (d, n) is stored as n/b contiguous blocks
   W_k = W[:, k·b:(k+1)·b]. `W @ x` is one `np.matmul` over the blocks (batched float32 gemv)
   plus one `np.add.reduce` over the n/b partial results. Each output is still `sum_j W[i,j]·x[j]` over the same
   float32 weights; only the grouping of the sum changes (blocks of b, which BLAS does internally anyway).
   I picked b = 128 with an in-place sweep (`prof/blocksize.py`): ms/token at positions 80 / 599 were 64: 8.48/9.75,
   **128: 8.35/9.64**, 256: 8.45/9.72, 384: 11.06/12.38. For other models, `block_size()` falls back to 256/64/96/192/32,
   or to one block.
   The shared embedding table is read back out of the blocked classifier, so there is still only one copy of it.
2. **Loader streams the file once** in file order through a 4 MB scratch buffer (`readinto`) and copies
   each chunk into its blocked place. Fresh-process load: plain `fromfile` 0.055 s; memmap + reorganise
   0.098 s; chunked readinto 0.075 s. Time to READY: 0.113 s (round 2) → 0.136–0.145 s (+~25 ms, paid back
   after ~12 tokens).
3. **Two V-cache layouts, picked per step.** V is stored as (t, d) and as (d, t) (one extra 768-float
   write per layer, +37 MB). The weighted sum uses einsum on (t, d) for T < 768 and vecdot on (d, t)
   for T ≥ 768, the measured crossover. Both sum over t in float32 in order, like the reference loop.
4. **Fused matvecs.** Since the loader copies everything anyway, [wq; wk; wv] and [w1; w3] are stacked
   along the output rows at load at no extra cost: one blocked matvec each instead of three and two
   (`prof/fuse2.py`: 4.09 → 3.97 ms for those matrices; in place −0.15 ms/token). K and V are then copied into the
   cache slots after RoPE.

### Round 3 results
- Weight matvecs: 10.0 ms → **7.4 ms/token (59 GB/s)**, which is about the plain-read speed of this machine
  (`ndarray.max` 56 GB/s). The rest of the forward (attention, norms, RoPE, Python) is ~0.95 ms.
- Per-token forward, min-of-N, interleaved with the round-2 snapshot (`prof/long.py`):

| position | 80 | 300 | 600 | 999 |
|---|---|---|---|---|
| round 2 (`run_v2`) | 10.62 | 11.23 | 12.15 | 13.23 |
| round 3 | **8.43** | **8.91** | **9.80** | **9.83** |

  That is −21 % at short context and −26 % near the context limit (round-3 block 128 is a further ~1 % on top of this table).
- End-to-end protocol (`prof/e2e.py`, best of 2 interleaved, noisy machine): GENERATE 64 ×3 prompts 2.52 → 2.19 s,
  LOGITS ×3 2.60 → 2.12 s, 1021-token prompt to context stop 13.06 → 10.54 s, total 18.71 → 15.12 s.
- Against round 1 the forward pass is now ~40 % faster at position 80 (13.5–14.9 → 8.35–8.45 ms/token).

### Round 3 tests
- check.py: quick PASS; full (3 prompts × 64 tokens) PASS, max logit error ≤ 2.7e-5 (tolerance 5e-4).
- `prof/verify.py` compares run.py with the plain round-1 implementation (`np.dot` per matrix, no blocking):
  real model over all 1024 positions with random ids (both weighted-sum branches, plus a reset and re-run), max
  |Δlogit| 1.4e-5, 0 argmax mismatches. Three synthetic random checkpoints also agree (≤ 2.3e-5, 0 mismatches):
  GQA with an unshared classifier (dim 128, 8 heads / 2 KV heads), dims that use 96-wide blocks (dim 288, hidden 768),
  and MQA with dims that fit no block size (dim 120, hidden 200, 1 KV head).
- Edge commands (PIECE, empty FIXED, GENERATE 0, empty prompt, empty ENCODE, leading space): output
  byte-identical to round 1. GENERATE 2000 still ends `DONE 238 stop`; the 1021-token prompt still ends `DONE 4 context`.

### Round 3: tried and rejected
- Attention scores via vecdot / einsum / per-head gemv, and K stored transposed: all slower than the batched
  matmul (`prof/attn2.py`, `prof/attn_split.py`).
- Padding the transposed-V row stride (16/32/64 floats): no gain.
- Block sizes 192/384 (slow gemv path or no clean split for 2048) and 64 (slightly slower than 128).
- Zero-copy strided block views (34 GB/s) and memmap-based reorganisation (slower load than chunked readinto).
- Still not done on purpose: batched prompt prefill (see round 2) and anything using threads.

## What is implemented (run.py)
- Checkpoint: 7-int header; negative vocab_size => unshared classifier; weights laid out exactly as
  `memory_map_weights` (including skipping the legacy freq_cis tables). Streamed once and stored as float32
  column blocks (QKV and W1|W3 stacked per layer); the shared embedding rows are read from the blocked classifier.
- Forward pass per token, per position, batch 1, all layers, float32 throughout (dtype audit in `prof/parts2.py`):
  rmsnorm (1e-5, `w*(ss*x)`), blocked float32 matvecs (batched gemv + partial-sum reduce), RoPE (`1/powf(10000, hd/hs)`, `pos*freq`, float32 cos/sin),
  attention over 0..pos with `score / sqrtf(head_size)`, max-subtracted softmax, GQA-aware head grouping,
  SwiGLU `x*(1/(1+exp(-x)))*w3x`, residuals, final rmsnorm, full logits every step.
- KV cache: float32. K is (layer, kv_head, seq_len, head_size); V is kept as (…, seq_len, head_size) and
  (…, head_size, seq_len) with identical contents. Each command zeroes the positions written by the
  previous command and starts at position 0.
- Tokenizer: byte-exact `encode` (dummy prefix " ", UTF-8 codepoint grouping capped at 4 bytes, byte fallback
  +3, greedy highest-score merges) and `decode` (strip leading space after BOS, `<0xNN>` -> raw byte).
- Protocol: stdin read as raw bytes, only the trailing `\n` removed. GENERATE forces prompt tokens, then greedy
  `np.argmax` (lowest index wins ties); `stop` on token 1 (not printed), `max` after max_new (checked first),
  `context` once seq_len positions have been forwarded (the token from position seq_len-1 is still emitted).
- No threads started, no thread counts set, no fast-math, no float64 in the model computation.

## Remaining limitations
- Still memory-bandwidth bound: ~7.4 ms/token streams 438 MB of float32 weights at ~59 GB/s, about this
  machine's single-thread read speed. Further per-token gains would have to come from the remaining ~0.95 ms
  (cold-cache attention ~0.55 ms at short context, ~25 small NumPy calls per layer).
- The speedups rely on measured properties of this machine's Accelerate build: the short-row gemv fast path,
  block size 128, and the V-layout switch at T = 768. On a different BLAS or CPU these choices may not be optimal
  (results would still be correct).
- Memory: +37 MB for the second V-cache layout. Startup is ~25 ms slower than round 2 because of the block reorganisation.
- Timings are noisy on this machine (up to ±30 % between runs); figures above are best-of or interleaved.
- `PIECE` for a piece that decodes to nothing (e.g. `<0x00>`, or " " after BOS) prints `HEX ` with an empty
  hex string (trailing space). INTERFACE.md does not specify this.
- Summation order differs from the C scalar loops (row dots via NumPy/Accelerate, pairwise sums), so logits
  differ from the reference by ~1e-5. Argmax near-ties could in principle flip.
- The merge floor is now float32(-1e10), as in C; round 1 used float64 -1e10. The two differ only for a score in
  (-1e10, -9999998976], and no score in this vocab falls there.
- Argmax with NaN logits follows NumPy (first NaN), not C's comparison semantics. NaN logits are not expected.
- FIXED/LOGITS with more ids than seq_len is not supported.
- Harness, fixtures, rules and reference were not modified. Only the supplied bundle and its public inputs were used.
