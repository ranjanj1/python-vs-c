# CB2 implementation notes

Technique: **CB2**. C with float32 arrays, plus Apple Accelerate calls. Source is `cb2.c`, built by `build.sh` into `./cb2`.
The only sources used were `reference/run.c`, `RULES.md`, `INTERFACE.md`, `check.py` and the fixtures.

## Commands run

```
cat RULES.md INTERFACE.md check.py; read reference/run.c; inspected fixtures.json and the model header
./build.sh            # cc -O3 -mcpu=native -Wall -Wextra -o cb2 cb2.c -framework Accelerate   (no fast-math)
python3 check.py -- ./cb2                          # PASS (max logit err ~1.1e-5, tol 5e-4)
python3 check.py --tokens 64 --prompts 3 -- ./cb2  # PASS (max logit err ~2.2e-5)
```
I also wrote ad-hoc scripts in /tmp/cb2t (not part of the deliverable):
- **PIECE:** checked against every `pieces` entry in fixtures.json (0 mismatches).
- **FIXED vs GENERATE:** FIXED argmax agrees with the GENERATE fixture ids.
- **Chunked prefill vs one-token-at-a-time:** compared over 600 positions, built with CHUNK=1. Max diff 3.3e-5.
- **Context stop:** a 1016-token prompt gives `DONE 9 context`; a prompt over 1024 tokens gives `DONE 0 context`.
- **Edge cases:** empty ENCODE, `GENERATE 0`, empty FIXED, and out-of-range ids.
- **Benchmarks:** `sgemv` vs `sgemm`(M=1) vs transposed-layout `sgemv`, and CHUNK sizes 64 to 1024.

## Design

- **Model and work are unchanged.** It uses the same float32 weights, architecture, RoPE, KV cache (layer, seq_len, kv_dim) and greedy argmax with the lowest index winning ties. Logits are computed at every forwarded position, including forced prompt positions in GENERATE, just as the reference does.
- **Weight layout (allowed by CB2).** At load, the checkpoint is `pread` into 16 KB-aligned buffers. Each layer gets one contiguous block: fused `[wq; wk; wv]`, then `wo`, then fused `[w1; w3]`, then `w2`. This turns QKV into one GEMM/GEMV and W1+W3 into one. The classifier shares the embedding when vocab_size > 0; otherwise `wcls` is read from after the skipped freq_cis block.
- **Known token sequences are forwarded in chunks** of up to 512 positions. This covers prompts, FIXED and LOGITS. Projections use `cblas_sgemm` (X·Wᵀ). Attention uses per-head `cblas_sgemm` for Q·Kᵀ over keys 0..p0+c-1, followed by a causal softmax per row: only keys 0..pos are normalised and the rest are set to 0. A second `cblas_sgemm` does att·V, reading K/V straight from the cache.
- **Decoding one token** uses `cblas_sgemv` for the projections and for per-head attention.
- **Element-wise work** uses plain C or vDSP/vForce: rmsnorm (`vDSP_svesq`), softmax (`vDSP_maxv`, `vvexpf`, `vDSP_sve`, `vDSP_vsdiv`), residual adds (`vDSP_vadd`), and SwiGLU and RoPE (scalar C, same formulas as the reference). There are no hand-written matrix loops and no intrinsics.
- **RoPE** cos/sin come from a table built at load. It uses the reference's exact float expressions (`powf`, `cosf`, `sinf`), so the values are bit-identical.
- **Single thread.** No threads are created and no thread counts are set; Accelerate's own threading is left to the harness environment.
- **Tokenizer** encode and decode are line-for-line ports of the reference, including byte fallback and the dummy-prefix behaviour.

## Interface choices where INTERFACE.md is silent

- **GENERATE stop order:** if the chosen token is 1, it ends with `stop`. Otherwise the token is emitted; then if count == max_new it ends with `max`. If the next forward position would be seq_len, it ends with `context`.
- **Prompt length:** a prompt longer than seq_len gives `DONE 0 context`. A prompt of exactly seq_len still emits one token, as the reference does.
- **`GENERATE 0 …`** answers `DONE 0 max` without running the model.
- **PIECE** prints `HEX ` followed by the bytes of the piece as a C string. An empty piece (for example `PIECE 1 29871`) gives `HEX ` with nothing after it. `<0x00>` also comes out empty, because the reference treats the piece as a C string.
- **Malformed commands,** out-of-range ids, or FIXED/LOGITS with more than seq_len ids get an `ERR …` line. These cases are not part of the contract.
- **KV cache between commands:** it is not re-zeroed. Every command writes positions 0..n-1 before reading them and never reads past the current position, so no state carries over from one command to the next.

## Results (this machine)

- Load to READY takes about 0.08–0.12 s.
- Decode runs at about 79 tok/s. That is about 34 GB/s of weight streaming, which a standalone `sgemv` benchmark shows is the single-core ceiling; `sgemm` with M=1 and a transposed layout gave the same speed.
- Prefill or FIXED for 1024 tokens takes about 0.42 s; 80 tokens takes about 0.04 s.

## Limitations

- Numerics differ from the reference's sequential sums because BLAS and vDSP add things up in a different order, and `vvexpf` is used in the attention softmax. The observed max logit error is about 2e-5, well inside the 5e-4 tolerance. On near-ties, greedy argmax could in principle pick a different token from the reference.
- Prefill computes the full Q·Kᵀ rectangle for each chunk and then masks it. This does a little extra work, never less.
- The tokenizer's merge loop is O(n²), as in the reference, so very long ENCODE/GENERATE texts are slow to tokenize.
- I only checked against the public fixtures; the final gate uses other inputs.
