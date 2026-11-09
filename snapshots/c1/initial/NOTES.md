# C1 port — notes

Technique set **C1**: C, compiler optimisation flags, CPU intrinsics (ARM NEON). No external math
library, no BLAS, no threads, no fast-math.

## Files added
| File | Purpose |
|---|---|
| `run_c1.c` | The implementation (single file). |
| `build.sh` | Build command. Produces `./run_c1`. |
| `tests/ref_encode.c` | Differential helper: `#include`s the unmodified `reference/run.c` (with `-DTESTING`) and prints its `encode()`/`decode()` output. |
| `tests/ref_logits.c` | Differential helper: unmodified reference `forward()`, writes all logits for an id sequence. |
| `tests/difftest.py` | Compares ENCODE (2009 strings, incl. random ASCII/UTF-8/invalid bytes, a 3000-char string) and PIECE (all 3×32000 `prev`∈{0,1,2} pairs) against the reference. |
| `tests/longctx.py` | Compares LOGITS against the reference over a full 1024-position sequence, checks FIXED argmaxes, GENERATE stop/`0`. |
| `tests/context.py` | Exercises the `context` stop (prompts of 981, 1006 and 1101 tokens). |
| `tests/bench.py` | Rough local timing only (the harness does the real timing). |

Nothing in the supplied bundle (harness, fixtures, reference, rules) was modified.

## Commands run
```
./build.sh          # cc -O3 -mcpu=apple-m1 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -o run_c1 run_c1.c
otool -L run_c1; nm -u run_c1      # only libSystem; no expf/sinf/cosf/powf/sqrtf imports
python3 check.py -- ./run_c1                          # PASS
python3 check.py --tokens 64 --prompts 3 -- ./run_c1  # PASS
cc -O2 -D_DARWIN_C_SOURCE -o tests/ref_encode tests/ref_encode.c
cc -O3 -D_DARWIN_C_SOURCE -o tests/ref_logits tests/ref_logits.c
python3 -I tests/difftest.py "$PWD"       # 98009 answers compared, 0 mismatches
python3 -I tests/longctx.py "$PWD" 1024   # worst max-abs logit error 4.8e-05 over 1024 positions; argmax 1024/1024
python3 -I tests/context.py "$PWD"        # DONE 44/19/0 context, as expected
python3 -I tests/bench.py "$PWD" ./run_c1
```
Matmul kernel variants (1/2/4/8 rows per pass, prefetch distances, 16- vs 64-byte loads) were
benchmarked from throw-away copies in `/tmp` (since deleted); the winner is in `run_c1.c`.

## Results (Apple M1, local, rough)
- `check.py` default and `--tokens 64 --prompts 3`: **PASS**, max logit error ≤ 2.9e-05 (tolerance 5e-4).
- Load to `READY`: ~0.3 s. Generation: ~8.7 ms per forward at short context (~50 GB/s of weight
  streaming, near one M1 core's bandwidth), ~12 ms per forward averaged over a 1024-position LOGITS run.
- Reference `run.c` (`-O3`, single thread) for the same 1024-position run: ~95 s vs ~12.5 s.

## What the implementation does
- **Same work as the reference:** the same checkpoint layout (including shared and unshared classifier
  through the negative vocab_size, plus the skipped legacy RoPE tables), float32 weights and activations,
  the same op order per layer, the same KV-cache layout and per-position writes, and the same greedy argmax
  (lowest index wins ties). Batch size 1 and **one token per forward pass**, positions 0..n-1. Prompt tokens
  are processed one at a time exactly like the reference: there is no multi-position "prefill" batching,
  because the rules say batch size 1 and the same per-step work.
- **Weights:** the file is read into a 128-byte-aligned heap buffer. The header is 28 bytes, so the
  in-place layout would not be 16-byte aligned. Weights stay resident between commands.
- **Matmul:** NEON, two rows per pass, `vld1q_f32_x4` 64-byte loads, 4 FMA accumulators per row.
  Has a scalar/tail fallback for shapes that are not multiples of 16.
- **Attention:** NEON dot products and a register-resident weighted V sum per head. Softmax and SiLU use a
  vectorised float32 `exp` (Cody–Waite reduction plus a degree-7 Taylor polynomial, with overflow,
  underflow and NaN handling).
- **No libm:** `sqrt` uses the `fsqrt` instruction (`__builtin_sqrtf`). RoPE cos/sin are precomputed at load
  into `seq_len × head_size/2` tables. The angle is formed exactly as in the reference, in float32
  (`freq = 1/pow(10000, hd/hs)`, `val = pos*freq`), and then evaluated with our own double-precision
  exp and sin/cos and rounded to float. Double is used only for this load-time table, not in the main
  computation.
- **Tokenizer:** the same algorithm as the reference: dummy prefix, UTF-8 codepoint grouping, byte
  fallback, and greedy best-score pair merging with the first index winning ties and a `-1e10`
  threshold. Lookups use a hash table instead of bsearch. Pair scores are kept per position and only the
  two neighbours are recomputed after each merge. The result is identical, verified by `tests/difftest.py`.
- **Decode/PIECE:** the reference `decode()` logic, including the leading-space strip after BOS and
  `<0xXX>` byte tokens via the same `sscanf` pattern. The hex output covers the C-string bytes, the same as
  the reference's returned `char*`.
- **Protocol:** see INTERFACE.md. stdout is flushed after every line. GENERATE prints each `T` line as soon
  as its token exists. The stop order after each new token is: token 1 → `stop`, then `count == max_new`
  → `max`, then needing a forward at `pos == seq_len` → `context`.

## Limitations / judgement calls
- Floating-point results are not bit-identical to the reference. The reduction order differs (vector
  accumulators), FMA is used (clang's default `-ffp-contract=on`; no fast-math), and exp/sin/cos are our
  own implementations. The observed max-abs logit difference is ≤ 5e-5 over 1024 positions. All
  argmaxes agreed in the tests run here, but a near-tie in logits could in principle flip a greedy choice.
- The kernel is tuned for and uses NEON on arm64 (Apple M1). It will not build on x86.
- Edge inputs the interface does not define are handled defensively but arbitrarily:
  - Unknown commands reply `ERROR ...`.
  - Out-of-range ids in FIXED/LOGITS are replaced by 0. An out-of-range PIECE id prints `HEX ` with no bytes.
  - FIXED/LOGITS sequences longer than `seq_len` are truncated to `seq_len`.
  - `GENERATE 0 ...` answers `DONE 0 max` without running the model.
  - A piece whose decoded byte is 0x00 prints as empty (`HEX `), matching C-string semantics.
- `check.py` passing is necessary but not sufficient. The final gate uses other inputs I have not seen.
