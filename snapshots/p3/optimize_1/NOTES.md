# P3 — NumPy float32 port of reference/run.c

Program: `python3 run.py <checkpoint.bin> <tokenizer.bin>` (also executable as `./run.py`).
Profiling scripts live in `prof/` (`prof/run_v1.py` is a frozen copy of the first version, used as the baseline).

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

## What is implemented (run.py)
- Checkpoint: 7-int header; negative vocab_size => unshared classifier; weights laid out exactly as
  `memory_map_weights` (including skipping the legacy freq_cis tables). Loaded once with `np.fromfile` as float32.
- Forward pass per token, per position, batch 1, all layers, float32 throughout (dtype audit in `prof/parts2.py`):
  rmsnorm (1e-5, `w*(ss*x)`), separate float32 matvecs, RoPE (`1/powf(10000, hd/hs)`, `pos*freq`, float32 cos/sin),
  attention over 0..pos with `score / sqrtf(head_size)`, max-subtracted softmax, GQA-aware head grouping,
  SwiGLU `x*(1/(1+exp(-x)))*w3x`, residuals, final rmsnorm, full logits every step.
- KV cache: float32, (layer, kv_head, seq_len, head_size). Each command zeroes the positions written by
  the previous command and starts at position 0.
- Tokenizer: byte-exact `encode` (dummy prefix " ", UTF-8 codepoint grouping capped at 4 bytes, byte fallback
  +3, greedy highest-score merges) and `decode` (strip leading space after BOS, `<0xNN>` -> raw byte).
- Protocol: stdin read as raw bytes, only the trailing `\n` removed. GENERATE forces prompt tokens, then greedy
  `np.argmax` (lowest index wins ties); `stop` on token 1 (not printed), `max` after max_new (checked first),
  `context` once seq_len positions have been forwarded (the token from position seq_len-1 is still emitted).
- No threads started, no thread counts set, no fast-math, no float64 in the model computation.

## Remaining limitations
- Still memory-bandwidth bound: ~10.4 ms/token is spent streaming 438 MB of float32 weights at ~42 GB/s
  through NumPy/Accelerate kernels. Without compiled code or batching, I found nothing in NumPy that streams faster.
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
