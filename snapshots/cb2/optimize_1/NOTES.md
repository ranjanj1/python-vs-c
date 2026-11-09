# CB2 implementation notes

Technique: **CB2**. C with float32 arrays, plus Apple Accelerate calls. Source is `cb2.c`, built by `build.sh` into `./cb2`.
The only sources used were `reference/run.c`, `RULES.md`, `INTERFACE.md`, `check.py` and the fixtures. I did not look at any other port, did not use the network, and did not open `/Users/Shared/pvc-kit`.
The machine is an Apple M1 with 8 GB RAM.

## Build and check

```
./build.sh            # cc -O3 -mcpu=native -Wall -Wextra -o cb2 cb2.c -framework Accelerate   (no fast-math)
python3 check.py -- ./cb2                          # PASS (max logit err ~1.2e-5, tol 5e-4)
python3 check.py --tokens 64 --prompts 3 -- ./cb2  # PASS (max logit err ~2.6e-5)
```

Optional build flags (all off by default):
- `-DPROF` adds a section profiler that prints a breakdown to stderr on QUIT.
- `-DBLOCK_MAX=n` sets the weight block width.
- `-DLOAD_PREAD` loads through a staging buffer instead of mmap.

---------------------------------------------------------------------------------------------------------

## Round 2: profiling and optimisation

### Development workloads used

The only workloads I used are the supplied fixtures; I did not invent or look for hidden inputs. `prof/dev.py` replays them through the normal stdin interface and times each phase from outside:
- ENCODE of every tokenizer fixture.
- GENERATE 64 for each of the 3 fixture prompts. It asserts the tokens match the fixture ids.
- LOGITS on each fixture `seq` (69/79/82 ids), written to a temp file.
- FIXED on the same sequences.

To check scaling I also ran synthetic long inputs: a 901-token prompt (`"dog " * 900`) with GENERATE 100, and a FIXED of 1024 ids (`prof/longgen.py`). I used these only to see how cost grows with context, not to tune toward particular outputs.

Timings on this machine drift by up to about 10% between runs (thermal state and other load). For comparisons, `prof/ab.py` alternates the binaries over several runs and reports medians.

### Profiling evidence

**1. Baseline (v1) section profile, GENERATE 3×64** (`prof/profrun.sh`, the `-DPROF` build):

| section | ms | share | effective weight bandwidth |
|---|---|---|---|
| qkv sgemv | 451 | 18.8% | 36 GB/s |
| wo | 152 | 6.3% | 36 GB/s |
| w1+w3 | 787 | 32.7% | 37 GB/s |
| w2 | 445 | 18.5% | 33 GB/s |
| classifier | 505 | 21.0% | 37 GB/s |
| attention | 38 | 1.6% | – |
| rope, swiglu, other | 27 | 1.1% | – |

Streaming the weights accounts for 97% of the time; decode is purely memory-bound.

**2. Bandwidth ceiling** (`prof/bw.c`, `prof/sweep.c`, `prof/sweep2.c`):
- A pure read (`vDSP_sve` over 440 MB) runs at **63 GB/s**, but `cblas_sgemv` NoTrans on the original row-major layout only reaches 25–35 GB/s for every model shape. So v1's note that 34 GB/s was the ceiling was **wrong**.
- `sgemv`-NoTrans, `sgemm` with M=1, and `vDSP_mmul` all give the same ~35 GB/s.
- `sgemv`-**Trans** on a weight block stored as contiguous Wᵀ (n × w) runs at **~63 GB/s whenever the block width w is between about 256 and 384**, for both n=768 and n=2048. Wider blocks fall to 25–45 GB/s and narrower ones to about 20 GB/s.
- Across repeated sweeps only 320–384 was fast every time. In A/B runs of the full program, block caps of 256/320/384/448 gave identical decode speed. Cap 256 made prefill about 17% faster (LOGITS 98 vs 119 ms), so I chose 256.

**3. Prefill GEMM** (`prof/gemm.c`, M=76): `sgemm` over the 256–384-wide transposed blocks reaches about 570 GFLOP/s, the same as the original layout (586). A single full-width transposed matrix is much worse (237). So the decode-friendly layout costs prefill nothing.

**4. Prefill profile** (fixture FIXED ×15, v1): weight GEMMs were 84% of the time and SwiGLU 9%, because SwiGLU called scalar `expf` per element. For FIXED 1024, causal attention was 25%, dominated by the softmax exps (about 75M `exp` calls, all of which the reference also performs).

**5. Long-context decode** (`prof/attn.c`, `prof/attn2.c`): at about 950 keys, per-head strided `sgemv` attention costs about 4 ms per token, while the KV data alone (about 70 MB) needs about 1.2 ms. Per token, over 12 layers:

| keys | per-head sgemv | block-diagonal sgemm | head-major sgemv |
|---|---|---|---|
| 8 | 0.04 ms | 0.11 ms | 0.06 ms |
| 100 | 0.17 ms | 0.19 ms | 0.29 ms |
| 300 | 1.16 ms | 0.92 ms | 0.93 ms |
| 1000 | 4.0 ms | 3.0 ms | 2.8 ms |

Narrow-N `sgemm` (N=12) is the limiting factor, at about 2 ms of the 3 ms.

**6. Tokenizer:** ENCODE of a 900-word text took **1608 ms**, more than the whole forward pass for that GENERATE. The reference merge loop re-runs `sprintf`+`bsearch` on every pair at every merge, which is O(n²) lookups.

### Changes in round 2

1. **Transposed, column-blocked weights.** Each matrix W (d×n) is stored as consecutive blocks; block b holds W[j0:j0+w,:]ᵀ as a contiguous n×w array, with w ≤ 256. Decode issues one `cblas_sgemv`(Trans) per block. Prefill issues one `cblas_sgemm`(NoTrans, NoTrans) per block, writing into columns j0.. of Y (ldc=d). Grouping the work this way is explicitly allowed by CB2. Every matrix product still goes through Accelerate, and the arithmetic is the same.
2. **Q/K/V and W1/W3 are now separate blocked matrices**, stored contiguously per layer in use order.
   - K and V are written by BLAS **directly into the KV cache rows** for the current chunk; RoPE is then applied in place, so the copies are gone.
   - W1 and W3 write into separate buffers, so **SwiGLU is a few whole-buffer vDSP/vForce calls**: `vDSP_vneg`, `vvexpf`, `vDSP_vsadd`, `vDSP_svdiv` (exactly 1/(1+e)), then two `vDSP_vmul`. The operation order matches the reference: `val = h1 * (1/(1+exp(-h1))); val *= h3`.
3. **Shared embedding is gathered from the transposed classifier blocks** (`cblas_scopy` with stride w), so the weights are held only once. An unshared checkpoint keeps a row-major embedding copy as well.
4. **Load: transpose straight from a read-only mmap** of the checkpoint (`vDSP_mtrans` per block), with no staging copy: 114 → 103 ms median against the pread+staging variant. `madvise` hints made no measurable difference, so they are not used.
5. **Causal row groups in chunked attention.** Each group of 64 query rows only multiplies against keys 0..p0+r1-1, instead of the whole chunk's rectangle. It still computes at least everything the reference computes.
6. **Long-context single-token attention.** From 160 keys upward, all heads go through two `sgemm` calls per layer:
   - Kᵀ-scores: K (T×kv_dim) @ Q_blk, where Q_blk (kv_dim × n_heads) is block-diagonal: column h holds q_h on its kv head's rows and exact zeros elsewhere.
   - Output: att (n_heads×T) @ V, keeping each head's diagonal block.

   Below 160 keys the per-head `sgemv` path is kept, because it is faster there. This only adds multiply-by-zero and discarded outputs; none of the reference's work is removed.
7. **Tokenizer merge loop with cached pair candidates.** Each adjacent pair's (merge id, score) is cached. After a merge, only the two neighbouring pairs are looked up again. The selection scan is unchanged: highest score, strict `>`, lowest index on ties, best_score starting at -1e10. The output is identical to the reference algorithm.

### Results: v1 vs v2

Medians of 5 alternating runs of `prof/ab.py`:

| workload | v1 | v2 |
|---|---|---|
| GENERATE 3×64 (fixture prompts) | 2415–2422 ms (≈79.5 tok/s) | 1450–1585 ms (≈121–132 tok/s) |
| LOGITS ×3 (fixture seqs) | 127–131 ms | 97–100 ms |
| FIXED ×3 (fixture seqs) | 124–128 ms | 92–97 ms |
| load to READY | 73–74 ms | 103–105 ms |
| GENERATE 100 after a 901-token prompt | 3474–3705 ms | 1324 ms |
| FIXED 1024 ids | 422–438 ms | 389–404 ms |
| ENCODE 900-word text | 1608 ms | 7.8 ms |

The v2 section profile for GENERATE 3×64 shows each weight GEMV at 57–60 GB/s, close to the 63 GB/s pure-read ceiling. Weight streaming is still about 96% of decode time; attention is 2.7%, SwiGLU 0.4%, and rope/other under 1%.

### Tests run in round 2 (all passing)

- `python3 check.py -- ./cb2` and `python3 check.py --tokens 64 --prompts 3 -- ./cb2`: PASS.
- `prof/dev.py`: asserts GENERATE 64 equals the fixture ids for all 3 prompts on every timed run.
- `prof/t.py`: PIECE matches all fixture `pieces` (0 mismatches). FIXED argmax agrees with the GENERATE ids. Edge cases (empty ENCODE, `GENERATE 0`, empty or out-of-range FIXED) and QUIT exit 0 also pass.
- `prof/cmp2.py`: LOGITS over 600 positions compared with the v1 sequential build (CHUNK=1).
  - New sequential build (which uses the block-diagonal path past 160 keys): max diff 3.5e-5.
  - New chunked build: max diff 3.7e-5.
- `prof/ctx.py`: a 1016-token prompt gives `DONE 9 context`; a 1031-token prompt gives `DONE 0 context`.
- `prof/enc.py`: the new ENCODE matches v1 (the verbatim reference algorithm) on 514 texts: fixtures, random word mixes, random Unicode including 4-byte UTF-8, and long texts. 0 mismatches.
- **Unshared-classifier path:** I derived a checkpoint in /tmp by negating vocab_size and appending the embedding as an explicit `wcls`. Its LOGITS are bit-identical to the shared model's and its GENERATE tokens are identical. The file was deleted afterwards.

### Remaining limitations

- **Decode is memory-bound** at about 57–60 GB/s against a measured ~63 GB/s single-core read ceiling. Without less weight traffic (quantisation, skipping the classifier on prompt positions, and so on, which the rules forbid) little headroom is left. Logits are still computed for every forced prompt position.
- **Load is about 30 ms slower than v1** (≈105 ms) because the weights are transposed once at startup. Whole workloads gain far more than that.
- **Long-context attention** is still about 2.5× its memory floor, because Accelerate's narrow-N/M `sgemm` is inefficient there. At 1024-token prefill the softmax exps (work the reference also does) dominate attention.
- **Block width.** Tuned on this M1 with microbenchmarks that vary ±10% between runs; the best width may differ on other chips.
- **Numerics** differ slightly from the reference's sequential sums (BLAS/vDSP summation order, `vvexpf` in softmax and SwiGLU). The observed max logit error is ≤2.6e-5 (tolerance 5e-4). On a near-tie, greedy argmax could in principle differ from the reference.
- **Scope of testing.** I only checked the public fixtures plus the self-made consistency tests above; the final gate uses other inputs.

---------------------------------------------------------------------------------------------------------

## Design (current)

- **Model and work are unchanged.** It uses the same float32 weights, architecture, RoPE, KV cache (layer, seq_len, kv_dim) and greedy argmax with the lowest index winning ties. Logits are computed at every forwarded position, including forced prompt positions in GENERATE.
- **Known token sequences are forwarded in chunks** of up to 512 positions (prompts, FIXED, LOGITS), using blocked `sgemm`. Single tokens use blocked `sgemv`.
- **Element-wise work** uses vDSP/vForce or plain scalar C:
  - rmsnorm: `vDSP_svesq`
  - softmax: `vDSP_maxv`, `vvexpf`, `vDSP_sve`, `vDSP_vsdiv`
  - SwiGLU: vDSP/vForce, as described in change 2
  - RoPE: scalar C, with a cos/sin table built from the reference's exact float expressions

  There are no hand-written matrix loops and no CPU intrinsics.
- **Single thread.** No threads are created and no thread counts are set.

## Interface choices where INTERFACE.md is silent (unchanged from v1)

- **GENERATE stop order:** if the chosen token is 1, it ends with `stop`. Otherwise the token is emitted; then if count == max_new it ends with `max`. If the next forward position would be seq_len, it ends with `context`.
- **Prompt length:** a prompt longer than seq_len gives `DONE 0 context`. A prompt of exactly seq_len emits one token, as the reference does.
- **`GENERATE 0`** answers `DONE 0 max` without running the model.
- **PIECE** prints `HEX ` followed by the bytes of the piece as a C string. An empty piece (for example `PIECE 1 29871`, or `<0x00>`) gives `HEX ` with nothing after it.
- **Malformed commands,** out-of-range ids, and FIXED/LOGITS with more than seq_len ids get an `ERR …` line.
- **KV cache between commands:** it is not re-zeroed. Every command writes positions 0..n-1 before reading them and never reads past the current position.

## v1 history (round 1)

v1 kept the row-major layout with fused QKV and W1/W3 weights, and used `sgemv`/`sgemm` NoTrans. It passed the same checks and ran at ≈79 tok/s decode.
