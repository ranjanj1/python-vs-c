# P4 — Numba port of reference/run.c

Program: `python3 run.py <checkpoint.bin> <tokenizer.bin>` (protocol per INTERFACE.md).
Check: `python3 check.py -- python3 run.py`.

## Result

- `python3 check.py -- python3 run.py` → **PASS** (max logit error 9.8e-6, tolerance 5e-4).
- `python3 check.py --tokens 64 --prompts 3 -- python3 run.py` → **PASS** (all 3 prompts, 64 tokens,
  max logit error 2.4e-5).
- Speed on this machine (noisy, varied between runs): ~85–100 tok/s for GENERATE with a short prompt;
  startup to `READY` ~1.4 s with a warm Numba cache (~2.8 s on the first run, when it compiles).

## What was built (`run.py`)

- **Model loading**: reads the 7-int header; a negative vocab_size means unshared classifier
  weights. Reads the float32 weights in the run.c order and skips the two legacy freq_cis blocks.
  Weight matrices are copied once at load into a transposed layout `(layer, in, out)` by a compiled
  blocked transpose. The values are unchanged float32; only the memory layout differs. The
  classifier gets its own transposed copy (shared with the embedding in this checkpoint). The
  embedding table is also kept in its original row layout for the token lookup.
- **Forward pass** (`forward`, all `@njit` nopython, `cache=True`, no `fastmath`, no `parallel`):
  - same ops as run.c, in the same order: rmsnorm → q/k/v → RoPE (powf/cosf/sinf per pair, as in
    run.c) → attention with softmax over 0..pos → wo → residual → rmsnorm → w1/w3 → SwiGLU → w2 →
    residual → final rmsnorm → classifier. The classifier runs at every position, forced prompt
    positions included, as in run.c.
  - **matvec** `matmul_t`/`matmul_t4`: `out[i] += Wt[j,i]*x[j]` with the loop over outputs `i` on
    the inside. This vectorizes without fastmath because no reduction is reordered, and each output
    is summed over `j` in ascending order, the same order as run.c's `matmul`. The two kernels
    differ only in how many `j` rows they take per pass (2 vs 4) and give bit-identical results.
    No BLAS, no `np.dot`, no `@`.
  - **KV cache**: per layer, keys are stored as `(n_kv_heads, head_size, seq_len)` (transposed),
    so the q·k scores vectorize over `t` while each score is still summed over `i` in order.
    Values are stored `(seq_len, kv_dim)`. Each score is divided by `sqrtf(head_size)` as in run.c.
    The cache is zeroed at the start of every command, and positions restart at 0.
  - Checked with `inspect_types` and the LLVM IR: no float64 variables or double arithmetic and no
    fast-math flags in the kernels. The only float64 values are literal constants immediately
    cast with `np.float32(...)`.
- **Tokenizer** (plain Python, not hot): mirrors run.c `encode`. BOS first, a dummy-prefix `" "`
  only for non-empty text, UTF-8 codepoint grouping with the `str_len < 4` limit, byte fallback
  `+3`, then greedy merges picking the highest score, with the earliest pair winning ties (strict
  `>`). Pair scores are cached and only a merge's neighbours are recomputed, which picks the same
  merges as run.c's full rescan. C-string semantics are kept: input is cut at a NUL, and `<0x00>`
  decodes to an empty piece. `decode`/PIECE strips the leading space after BOS and maps
  `<0xHH>` to the raw byte.
- **Protocol**: GENERATE is greedy argmax (lowest index wins ties). Prompt tokens are forced and
  only the chosen tokens are printed, each flushed as soon as it is chosen. The stop reason is
  `stop` if the chosen token is 1, `max` after max_new tokens, and `context` once seq_len positions
  are used. As in run.c's loop, the token chosen at the last position (seq_len-1) is still
  emitted. FIXED/LOGITS run full logits at every step; LOGITS writes n×vocab little-endian float32.
- One compute thread: Numba is used without `parallel`, and no threads are started or configured.

## Commands run

```
cat RULES.md INTERFACE.md check.py reference/run.c; head fixtures.json
python3 -c "import numba, numpy; print(numba.__version__, numpy.__version__)"   # 0.68.1 2.5.4
python3 -c "<read model header>"            # (768, 2048, 12, 12, 12, 32000, 1024)
python3 -c "<scan tokenizer.bin>"           # no duplicate pieces, 256 well-formed <0xHH> tokens
python3 scratch/bench_mm.py                 # dot-product kernels: lane-split versions don't vectorize w/o fastmath
python3 scratch/bench_mm2.py                # transposed axpy form ~3x faster than best dot form
python3 scratch/bench_mm3.py                # unroll/tiling variants (tiling was slower)
cd scratch && PYTHONPATH=.. python3 bench_unroll.py   # 2-row unroll best for layer mats, 4-row for classifier
python3 check.py -- python3 run.py
python3 check.py --tokens 64 --prompts 3 -- python3 run.py
NUMBA_CACHE_DIR=$(mktemp -d) python3 - <<EOF ... inspect_types / inspect_llvm ... EOF   # float32-only, no fast flags
python3 scratch/speed.py                    # startup + tok/s
python3 scratch/edge.py                     # PIECE vs fixtures, empty prompt, max_new 0, FIXED, context stop
```

`scratch/` holds only these benchmark and test helpers; `run.py` does not use them.

## Limitations / choices to note

- **Summation order**: each matvec output is summed in the same order as run.c. Numba emits no
  FMA here (no `contract` flag), while the build that produced the fixtures may have fused
  multiply-adds. The differences are ~1e-5 in the logits.
- **Load-time transpose**: weights sit in memory once, in transposed form, plus an extra 98 MB
  transposed classifier copy because the classifier is shared with the embedding. Peak memory
  during load is about 1 GB.
- **Numba on-disk cache** (`cache=True`) writes compiled code to `__pycache__/` beside `run.py`.
  Without it (e.g. a read-only dir with no fallback), the first start compiles for a few seconds.
  Before `READY`, a single warm-up forward runs and the KV cache is reset afterwards.
- **Unspecified edge cases**, handled as follows: an empty decoded piece prints `HEX ` (trailing
  space, no bytes); FIXED/LOGITS with more than seq_len ids processes only the first seq_len;
  `GENERATE 0 ...` answers `DONE 0 max` without a forward pass; if `max` and `context` happen on
  the same token, `max` is reported; unknown commands answer `ERR unknown command`. Token ids are
  not range-checked (bounds checks are off in the kernels).
- The tokenizer and protocol handling are plain Python (not compute); only the arithmetic is
  compiled.
- Throughput on this machine moved a lot between runs (roughly 62–104 tok/s for the same command),
  apparently from other load, so the speed figures above are rough.
