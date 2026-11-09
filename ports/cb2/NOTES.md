# CB2 implementation notes

Technique: **CB2**. C with float32 arrays, plus Apple Accelerate calls. Source is `cb2.c`, built by `build.sh` into `./cb2`.
The only sources used were `reference/run.c`, `RULES.md`, `INTERFACE.md`, `check.py` and the fixtures. I did not look at any other port, did not use the network, and did not open `/Users/Shared/pvc-kit`.
The machine is an Apple M1 with 8 GB RAM.

## Build and check

```
./build.sh            # cc -O3 -mcpu=native -Wall -Wextra -o cb2 cb2.c -framework Accelerate   (no fast-math)
python3 check.py -- ./cb2                          # PASS (max logit err ~1.2e-5, tol 5e-4)
python3 check.py --tokens 64 --prompts 3 -- ./cb2  # PASS (max logit err ~2.4e-5)
cc -O2 -o /tmp/cb2t/refdrv prof/refdrv.c -lm && python3 prof/cmpref.py 300   # vs supplied reference: max diff 3.1e-5, argmax 300/300, ENCODE 0 mismatches
```

Optional build flags (all off by default):
- `-DPROF` adds a section profiler that prints a breakdown to stderr on QUIT.
- `-DBLOCK_MAX=n` sets the weight block width.
- `-DLOAD_PREAD` loads through a staging buffer instead of mmap.

---------------------------------------------------------------------------------------------------------

## Round 4: batch-size-1 fix (failed rule check)

**The failure.** Rule 1 says "Batch size 1". Rounds 1–3 had `forward_chunk`, which forwarded up to 512 positions in one pass using `cblas_sgemm`. It was used for GENERATE prompts, FIXED and LOGITS, so it violated the rule. As a result, the LOGITS/FIXED/prefill timings reported for rounds 1–3 below are **not valid results**.

**The fix** (`cb2.c`):
- `forward_chunk` is replaced by `forward(token, pos)`, one complete forward pass for one position.
  - Every weight product is one `cblas_sgemv` per weight block, with no `sgemm` across positions.
  - That position's K/V are written straight into its KV-cache row.
  - The logits go to a buffer for one position.
- `run_fixed` (FIXED/LOGITS) calls `forward(ids[pos], pos)` for pos = 0..n-1 in order, taking the argmax and/or writing that position's logits after each pass.
- GENERATE forwards every forced prompt token one at a time, then each new token, exactly like the reference's `generate` loop.
- Removed: the `CHUNK` buffers, the multi-row causal attention (`ATT_ROWS`), the multi-row `sgemm` branch of `matmul`, and the SwiGLU tiling, which was only useful for multi-row buffers.
- The only `cblas_sgemm` left is in the long-context attention (≥160 keys) of a single position. It multiplies that one query position's heads (block-diagonal queries) against the cached K and V. No other positions take part.

**Block width.** Weights are now used only by `sgemv`, so I re-ran the A/B (`prof/ab.py`, 4 alternating runs) with caps of 256/384/512. They came out equal within noise: GENERATE 3×64 took 1681/1686/1677 ms and LOGITS ×3 took 1702/1706/1700 ms. The cap stays at 256.

**Tests (all passing).**
- `check.py` default and `--tokens 64 --prompts 3`: PASS, max logit error 2.4e-5.
- New `prof/refdrv.c` + `prof/cmpref.py`: a test driver that includes the **supplied reference `reference/run.c` unmodified** (via its `TESTING` guard). The program is compared directly against the reference's own `forward()` and `encode()`:
  - LOGITS over 300 positions: max abs diff 3.1e-5, argmax agreement 300/300.
  - ENCODE: 313 texts (fixtures, random words and Unicode, long texts), 0 mismatches.
- `prof/t.py`: PIECE fixtures match (0 mismatches), FIXED/GENERATE are consistent, edge cases pass, exit 0.
- `prof/ctx.py`: `DONE 9 context` and `DONE 0 context` as expected.

**Results with batch size 1** (fixture workloads, `prof/dev.py`; medians, M1 under memory pressure):

| workload | round 3 (chunked, invalid) | round 4 (batch 1) |
|---|---|---|
| GENERATE 3×64 | 1454 ms | 1672–1686 ms |
| LOGITS ×3 (69/79/82 positions) | 95 ms | 1695–1706 ms |
| FIXED ×3 | 92 ms | 1690–1699 ms |
| FIXED 1024 ids | 387 ms | 9.26 s |
| time to first token (5/15/18-token prompts) | 9 / 11 / 14 ms | 36 / 107 / 128 ms |
| decode step | 7.3 ms | 7.2–7.3 ms |
| load to READY | 95 ms | 98–112 ms (memory-pressure outliers up to 380 ms) |

GENERATE is slower because its forced prompt tokens (5 + 15 + 18) now each cost a full step.

**Remaining limitations after round 4.**
- **Every position costs one full weight stream** of about 438 MB at about 59 GB/s, roughly 7.3 ms on this M1. That is the floor for batch size 1 with float32 weights. LOGITS, FIXED and prompt processing now scale linearly at that rate.
- The round-2 and round-3 analyses of decode (blocked transposed `sgemv` at the bandwidth ceiling, hybrid attention, tokenizer, load) still apply. The prefill-GEMM analyses no longer apply.
- All earlier limitations (numerics within 3.1e-5 of the reference, load page faults, memory pressure on this machine) still apply.

---------------------------------------------------------------------------------------------------------

## Round 3: profiling and optimisation

*(Historical. The chunked prefill described in rounds 1–3 violated rule 1 and was removed in round 4. Prefill, LOGITS and FIXED figures in these sections are not valid results.)*

The workloads are the same supplied fixtures, run with `prof/dev.py` and `prof/ab.py` (alternating binaries, medians) as in round 2. I also used `prof/ttft.py` to measure time to first token against decode step time for the 3 fixture prompts.

### Environment finding

The machine (M1, 8 GB) was under heavy memory pressure:
- `memory_pressure`/`vm_stat` showed about 160–300 MB free and about 3.2 GB in the compressor, with tens of millions of swap-ins and swap-outs since boot.
- model.bin sometimes dropped out of the page cache between runs; one load took 898 ms instead of about 95 ms.
- Run-to-run speed drifted by about ±10%.

So every comparison below uses alternating runs and medians, and I favoured changes that are robust and lower memory use.

### Profiling evidence

**1. Decode is at the memory ceiling.**
- Section profile (`-DPROF`, GENERATE 3×64): weight GEMVs are about 96% of time, attention 2.6%, everything else under 1%.
- `prof/ovh.c`: pure read ceiling 57.5–59.3 GB/s today; a stream of 256-wide transposed `sgemv` calls runs at 57.4–59.3 GB/s, the same as a pure read.
- Per-call overhead is 0.57 µs (cached 64×256 block), about 0.28 ms/token in total, which overlaps with streaming.
- An in-program probe that streams every decode matrix back to back reached 60.5 GB/s at best, though repeats ranged from 35 to 60 GB/s (noise).
- Best decode step measured: 7.27–7.39 ms (`prof/ttft.py`), against 438 MB / 59 GB/s = 7.4 ms. **There's no meaningful headroom left in decode** without reading fewer bytes.

**2. Decode attention, cold cache** (`prof/attn3.c`). The KV cache is evicted by the weight stream before each layer's attention, so I re-ran the strategy comparison with a 96 MB flush between layers:

| keys | strided | block-diag | head-major | head-major Kᵀ |
|---|---|---|---|---|
| 8 | 0.08 | 0.30 | 0.19 | 0.47 |
| 48 | 0.28 | 0.34 | 0.32 | 0.33 |
| 120 | 0.64 | 0.56 | 0.57 | 0.66 |
| 300 | 1.50 | 1.25 | 1.09 | 1.37 |
| 1000 | 5.2 | 3.5 | 3.5 | 4.1 |

(ms per token.) Below about 100 keys the current strided per-head path is still best, and the switch at 160 keys is still about right. A head-major cache would gain about 12% only beyond 300 keys and would need scatter copies in prefill, so I didn't do it. At fixture lengths attention is latency-bound at 0.1–0.4 ms/token.

**3. Load** (`/tmp` load-phase timers, `prof/tr.c`, `prof/fault.c`). Warm load breaks down as:

| phase | time |
|---|---|
| setup | 6 ms |
| classifier transpose | 18–19 ms |
| layer transposes | 62 ms |
| KV memset + RoPE | 5.3 ms |
| tokenizer | 7.5 ms |

For the 98 MB classifier, pure `vDSP_mtrans` work is 9–11 ms, destination page faults add about 4.5 ms and mmap source faults about 3 ms.
- Superpages are unavailable: `mach_vm_allocate` with `VM_FLAGS_SUPERPAGE_SIZE_ANY` returns invalid argument on arm64 macOS.
- vecLib has no other out-of-place transpose, and `vDSP_mtrans` takes no leading dimension, so it can't be tiled differently.
- pread into a warm staging buffer plus transpose is slower (22–30 ms vs 17.6 ms).

**4. Prefill GEMM** (`prof/gemm2.c`, cold weights, K=768 and 2048):

| M | w=256 | w=512 | w=1024 |
|---|---|---|---|
| 16 | 355–396 | ≈180 | ≈155 |
| 76 | 690–720 | ≈450 | 370–470 |
| 128 | 820–840 | ≈620 | 430–570 |

(GFLOP/s.) This confirms 256 as the block width. Writing to a contiguous output (ldc=w) gains only about 5%, before the scatter it would then need. In the program, prefill GEMMs already run at about 660–700 GFLOP/s.

**5. Accelerate alternatives.** `BNNSMatMul` on the same blocks gives bit-identical results and is slower: 58.5 vs 63.2 GB/s at M=1, 646 vs 724 GFLOP/s at M=76 (`prof/bnns.c`). Rejected.

**6. Time to first token:** 9.4 / 11.0 / 14.4 ms for the 5/15/18-token fixture prompts, against a 7.3–7.4 ms decode step. Small-M `sgemm` (M=16: about 360 GFLOP/s) makes prompt prefill compute-bound. Logits for every forced prompt position are kept (same work).

### Changes in round 3

1. **KV cache as lazily zero-filled anonymous mappings** (`mmap(MAP_ANON)`), replacing posix_memalign + a 75 MB `memset`. Contents are guaranteed zero like the reference's calloc. Pages only become resident when positions are written, which saves load time and 75 MB of steady-state memory for short contexts.
2. **Tokenizer load: one `pread` of the whole file, with strings in a single arena,** replacing about 96k `fread` calls and 32k `malloc`s. Vocab contents, scores and the `qsort`/`bsearch` lookup are unchanged.
3. **SwiGLU in 2048-float tiles,** so the six vDSP/vForce passes stay in L1 during prefill. Same per-element operations and order. No measurable change at fixture sizes; kept because it is never worse.

### Results: v2 vs v3

Medians of 5 alternating runs (`prof/ab.py`):

| workload | v2 | v3 |
|---|---|---|
| load to READY | 103.3–104.5 ms | 95.3–95.5 ms |
| GENERATE 3×64 | 1445–1455 ms (≈132 tok/s) | 1448–1454 ms (≈132 tok/s) |
| LOGITS ×3 | 95–97 ms | 95–96 ms |
| FIXED ×3 | 92 ms | 92 ms |
| GENERATE 100 after a 901-token prompt / FIXED 1024 | 1323 / 389 ms | 1338 / 387 ms (noise) |

Peak RSS is unchanged at 882 MB (`/usr/bin/time -l`). The peak happens during load: the mapped checkpoint plus the transposed copy. Steady state is 75 MB lower until long contexts actually fill the KV cache.

### Tested and rejected in round 3

- **`BNNSMatMul`:** slower than BLAS at every M (above).
- **Head-major or Kᵀ-per-head KV layouts:** no gain at fixture lengths; small gains only beyond 300 keys (above).
- **Contiguous-output prefill GEMM plus scatter:** about 5% on the GEMM alone, before the scatter cost.
- **Staging-buffer load (`-DLOAD_PREAD`), `madvise`, superpages:** slower, no effect, and unavailable, respectively.
- **Speculative or lookahead decoding:** not attempted. Drafting tokens and verifying them in batches would change KV-cache write behaviour and add prompt-dependent shortcuts, which conflicts with RULES.md rule 1.

### Tests run in round 3 (all passing)

- `python3 check.py -- ./cb2` and `python3 check.py --tokens 64 --prompts 3 -- ./cb2`: PASS, max logit error 2.6e-5.
- `prof/dev.py`: asserts the GENERATE output equals the fixture ids on every timed run.
- `prof/t.py`: PIECE fixtures (0 mismatches), FIXED/GENERATE consistency, and edge cases all pass; exit 0.
- `prof/ctx.py`: `DONE 9 context` and `DONE 0 context` as expected.
- `prof/cmp2.py`, comparing against the v1 sequential build over 600 positions:
  - rebuilt sequential CHUNK=1 build: 3.5e-5
  - chunked build: 3.7e-5
- `prof/enc.py`: ENCODE matches v1 on 514 texts (0 mismatches), which covers the new tokenizer loader.

### Remaining limitations after round 3

- **Decode and prefill are at hardware or Accelerate kernel limits on this M1.** Decode streams weights at about 59 GB/s against a 59–63 GB/s pure-read ceiling. Prefill GEMMs run at about 700 GFLOP/s, Accelerate's best for these shapes and this layout. Further large gains would need fewer bytes or flops, which the rules forbid.
- **Load is about 95 ms:** roughly 45 ms of transposing plus about 35 ms of unavoidable page faults. A cold page cache (seen under this machine's memory pressure) can push it to about 0.9 s, and nothing in the program can prevent that.
- **Short-context decode attention** (0.1–0.4 ms/token) is latency-bound because the weight stream evicts the KV cache.
- All earlier limitations (numerics, testing scope) still apply.

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
- **Batch size 1 (round 4).** Every position, in every command, gets its own complete forward pass, in order: `forward(token, pos)`.
  - FIXED and LOGITS: positions 0..n-1.
  - GENERATE: each forced prompt token, then each new token.
- **Weight products:** one `cblas_sgemv`(Trans) per column block of the transposed, blocked weights. There is no multi-position `sgemm`.
- **Attention for the current position:**
  - Below 160 keys: per-head `cblas_sgemv`.
  - From 160 keys: two `cblas_sgemm` calls over that one position's heads (block-diagonal queries). This still involves only one query position.
- **Element-wise work** uses vDSP/vForce or plain scalar C:
  - rmsnorm: `vDSP_svesq`
  - softmax: `vDSP_maxv`, `vvexpf`, `vDSP_sve`, `vDSP_vsdiv`
  - SwiGLU: vDSP/vForce
  - RoPE: scalar C, with a cos/sin table built from the reference's exact float expressions

  There are no hand-written matrix loops and no CPU intrinsics.
- **Single thread.** No threads are created and no thread counts are set.
- **Memory.** The KV cache uses lazily zero-filled anonymous mappings, and tokenizer strings live in one arena.

## Interface choices where INTERFACE.md is silent (unchanged from v1)

- **GENERATE stop order:** if the chosen token is 1, it ends with `stop`. Otherwise the token is emitted; then if count == max_new it ends with `max`. If the next forward position would be seq_len, it ends with `context`.
- **Prompt length:** a prompt longer than seq_len gives `DONE 0 context`. A prompt of exactly seq_len emits one token, as the reference does.
- **`GENERATE 0`** answers `DONE 0 max` without running the model.
- **PIECE** prints `HEX ` followed by the bytes of the piece as a C string. An empty piece (for example `PIECE 1 29871`, or `<0x00>`) gives `HEX ` with nothing after it.
- **Malformed commands,** out-of-range ids, and FIXED/LOGITS with more than seq_len ids get an `ERR …` line.
- **KV cache between commands:** it is not re-zeroed. Every command writes positions 0..n-1 before reading them and never reads past the current position.

## v1 history (round 1)

v1 kept the row-major layout with fused QKV and W1/W3 weights, and used `sgemv`/`sgemm` NoTrans. It passed the same checks and ran at ≈79 tok/s decode.
