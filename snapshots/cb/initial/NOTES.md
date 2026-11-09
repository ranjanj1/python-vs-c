# CB port: C + Apple Accelerate BLAS

## Files
- `cb.c`: the implementation (single file). `cb` is the built binary.
- Nothing in the harness, fixtures, rules or `reference/` was changed.

## Build
```
clang -O2 -Wall -Wextra -o cb cb.c -framework Accelerate
```
No `-ffast-math` or `-Ofast`. The code starts no threads and sets no thread counts, so Accelerate runs with whatever the harness environment gives it.

## Commands run
```
cat RULES.md INTERFACE.md check.py reference/run.c         # read the spec
python3 -c "...json/struct..."                              # inspect fixtures.json + model header
clang -O2 -Wall -Wextra -o cb cb.c -framework Accelerate
python3 check.py -- ./cb                                    # PASS
python3 check.py --tokens 64 --prompts 3 -- ./cb            # PASS (max logit err ~3e-5, tol 5e-4)
python3 /tmp/cbtest/extra.py $PWD     # my own tests: PIECE vs fixture pieces, FIXED argmax, empty prompt,
                                      # GENERATE 0, stop on token 1, QUIT exit code -> all OK
python3 - (inline)                    # 691-token prompt + GENERATE 5000 -> DONE 334 context (= 1024-691+1)
python3 /tmp/cbtest/bench.py ...      # mmap vs aligned-copy weights, ~62 vs ~70 tok/s for 200 tokens
```
(The `/tmp/cbtest` scripts are throwaway test drivers. They are not part of the deliverable.)

## Model (from the checkpoint header)
dim 768, hidden 2048, 12 layers, 12 heads, 12 kv heads, vocab 32000 (positive, so the classifier shares the embedding weights), seq_len 1024.

## What matches the reference
- **Same weight layout, float32 throughout.** The weights are the same float32 values as `model.bin`. Only their location in memory changes (see Changes). Shared or unshared classifier is chosen from the sign of vocab_size.
- **Matmuls use the same shapes.** Each `matmul(W(d,n) @ x)` becomes one `cblas_sgemv(RowMajor, NoTrans, d, n, ...)`: q/k/v/o, w1/w3/w2 and the classifier, one token at a time (batch 1, no prompt batching into GEMM).
- **Attention uses BLAS matvecs.** Per head, scores are `K[0..pos] @ q` with sgemv (lda = kv_dim), then the reference's `/ sqrtf(head_size)` and softmax. The weighted value sum is `V[0..pos]^T @ att` with transposed sgemv.
- **Scalar code is copied from the reference:** rmsnorm, softmax, RoPE formula, SwiGLU, residuals and argmax (strict `>`, so the lowest index wins ties), with the same `expf/powf/cosf/sinf`.
- **The tokenizer is the reference's.** BPE encode is the reference algorithm, including its UTF-8 handling, byte fallback and dummy prefix. `PIECE` uses the reference `decode`: it strips the leading space after BOS and maps `<0xXX>` to the raw byte. It prints the bytes up to the C string terminator, so `<0x00>` gives `HEX ` with nothing after it.
- **GENERATE follows the reference loop.** Prompt tokens are forced, and each command starts at pos 0. Greedy choice. `stop` if the chosen token is 1, `max` once max_new tokens are printed, `context` when the position reaches seq_len. If max_new and context are reached on the same token, it reports `max`.
- **The KV cache is clean for every command:** before each GENERATE/FIXED/LOGITS, the rows the previous command wrote are zeroed.

## Changes that don't change results
1. **Weights copied to aligned memory.** At load, the weights are copied once into a page-aligned buffer instead of being read through mmap at byte offset 28. Same values, about 10% faster sgemv, no page faults on the first command.
2. **RoPE angles computed once per forward pass.** cos/sin are computed with the exact reference formula once per forward instead of once per layer. They don't depend on the layer, so the values are bit-identical.
3. **Vocab sorted at load.** The sorted vocab for encode is built at load instead of lazily on the first ENCODE.

## Limitations / notes
- **Results are not bit-exact with the reference.** Accelerate's sgemv sums in a different order from the reference's serial loop, so logits differ slightly (≤3e-5 on the public fixtures, tolerance 5e-4). An argmax near-tie could in principle flip on other inputs.
- **Accelerate threading is not controlled by the program.** It may use multiple threads internally unless the harness limits it (e.g. via `VECLIB_MAXIMUM_THREADS`). Per the rules, the program does not set this.
- **Inputs the interface doesn't define:**
  - FIXED/LOGITS with more than seq_len ids process only the first seq_len.
  - Out-of-range ids, or an unwritable LOGITS path, answer `ERR ...`.
  - Unknown commands answer `ERR unknown command`.
  - Blank lines are ignored.
- **Speed:** about 65–72 tok/s on this machine for GENERATE with default Accelerate threading.
