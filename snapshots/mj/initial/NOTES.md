# MJ — llama2.c in Mojo 1.2.0

Single source file `mj.mojo` → executable `mj`. It implements `reference/run.c` (forward pass, BPE
tokenizer, greedy argmax) behind the line protocol in `INTERFACE.md`. It uses only the Mojo 1.2.0
standard library: SIMD, compile-time parameters, `std.math`, file I/O and raw fd reads. It does not
use Python interop, MAX, `parallelize`, threads, fast-math or any external math library.

## Commands used

```
mojo --version                                   # Mojo 1.2.0 (2016d0cd)
mojo build mj.mojo -o mj                         # default -O3, no extra flags
python3 check.py -- ./mj                         # quick check: PASS
python3 check.py --tokens 64 --prompts 3 -- ./mj # full public check: PASS
```

Most recent full check: max logit error 1.1e-5 to 2.7e-5 (tolerance 5e-4). All tokenizer cases,
all three 64-token generations and all logit positions pass.

Extra checks I ran with my own scripts in `/tmp/mjtest` (outside the bundle; the harness was not
modified):
- every `pieces` entry in `fixtures.json` through `PIECE`: all match;
- `FIXED` over each fixture `seq`: the argmaxes reproduce the fixture continuations;
- several commands written to stdin in one write (pipelined): handled correctly;
- `GENERATE 0`, a `stop` ending (token 1 chosen), and a `context` ending (877-token prompt gives
  1024 − 877 + 1 = 148 new tokens, as in run.c's `while (pos < seq_len)` loop).

Speed on this machine, single thread: about 120 tok/s for short contexts and about 110 tok/s
averaged over a full 1024-position `FIXED`. A plain SIMD read of the 438 MB weight file measured
about 62 GB/s, which is about 7.1 ms or 141 tok/s, so the matmuls run at about 86% of the
single-core memory-bandwidth limit.

## What matches the reference

- **Model:** checkpoint header, weight layout, the skipped legacy `freq_cis` block and shared or
  unshared classifier (negative `vocab_size`) all follow `memory_map_weights`. The weights are
  float32, read into memory once at startup. Every layer runs on every token: no quantization and
  no skipped work.
- **Forward pass:** same order of operations as `forward()`: rmsnorm (1e-5, `1/sqrt`), q/k/v
  matmuls, RoPE on q and on k for `i < kv_dim`, attention over positions 0..pos with
  `score / sqrt(head_size)`, softmax, weighted sum of V, `wo`, residual, rmsnorm, `w1`/`w3`,
  SwiGLU `x*(1/(1+exp(-x)))*w3x`, `w2`, residual, final rmsnorm and classifier.
- **KV cache:** layout is `(layer, seq_len, kv_dim)`; k and v are written at `pos` and attention
  reads rows 0..pos. Grouped-query attention (`kv_mul`) is supported. Every command restarts at
  position 0. Attention at position p only reads cache rows 0..p, which the same command has just
  written, so the results are identical to clearing the cache. The cache is therefore not memset
  between commands.
- **Tokenizer:** byte-for-byte port of `encode`:
  - BOS;
  - the dummy-prefix `" "` token only for non-empty text;
  - UTF-8 grouping using the same `str_len < 4` rule;
  - byte fallback `+3`;
  - greedy best-score merges, where a strictly greater score wins, so the leftmost pair wins ties
    as in run.c;
  - text treated as a C string (it ends at the first NUL).

  The `bsearch` over sorted vocab is replaced by an exact-match hash table. The vocab has no
  duplicate strings, so lookups return the same ids.
- **PIECE:** port of `decode`:
  - after BOS, one leading space is stripped;
  - `<0xXX>` pieces become their raw byte;
  - the output is the lowercase hex of the C string. Byte `<0x00>` is an empty C string, so it
    gives `HEX ` with nothing after the space.
- **GENERATE:** prompt tokens are forced and new tokens come from greedy argmax, where the lowest
  index wins ties. If the next token is 1 the run ends with `stop`. As in run.c, this also applies
  to a forced prompt token, though a forced 1 essentially cannot happen. When both limits hit at
  once, `max` is checked before `context`. `T` lines are flushed as each token is produced.

## Implementation notes and optimisations

- **matmul:** blocks of 2 rows share each load of `x`, with a 32-lane float32 SIMD accumulator per
  row and explicit FMA. I benchmarked 1, 2, 3, 4 and 8 rows with 8 to 64 lanes; 2×32 was fastest.
  Software prefetch gave no gain. Tails handle any `n` and `d`.
- **Attention:** walks the cache by position, reading each K/V row (all heads) contiguously. This
  gave about 17% more speed at long context than the per-head strided walk. The head size is a
  compile-time parameter for 16, 32, 64 and 128; a runtime-sized generic path covers any other size.
  I tested that path by forcing it on, and it passes the check.
- **Vectorized element-wise steps:** rmsnorm, softmax, SwiGLU and the residual adds.
- **Summation order:** SIMD changes the order of float32 additions compared with the C scalar
  loops. These are ordinary rounding-level differences; the measured error is about 2e-5.
- **RoPE:** the reference recomputes `cos`/`sin(pos*freq)` in every layer. Here they are computed
  once per token and reused by all layers; the values are the same.
  - The frequency constants follow `1/powf(10000, head_dim/(float)head_size)`. The exponent is
    formed in float32 as in C.
  - The power is evaluated in float64 and rounded to float32. Mojo's float32 `**` is
    `exp(y*log(x))` in float32 and less accurate than libm `powf`. This affects only `head_size/2`
    constants computed at load time. The main computation stays float32.
- **exp:** `exp` comes from Mojo's standard library (polynomial). It can differ from libm `expf`
  by an ulp or so; `cos`/`sin` lower to the LLVM intrinsics.
- **stdin:** read through a raw file descriptor with my own line buffer.
  `input()`/`_fdopen().readline()` open a new buffered `FILE*` on every call and could drop lines
  when the harness pipelines commands.

## Limitations / deviations

- `FIXED`/`LOGITS` with more than `seq_len` ids process only the first `seq_len` and report that
  count (run.c cannot run past `seq_len`).
- Unknown commands get `ERR unknown command`; malformed numbers are parsed leniently.
- Weights are read with `read()` instead of `mmap`: about 0.4 GB of RAM, one-off load time before
  `READY`.
- The public check only exercises this model (dim 768, 12 layers, 12 heads, head size 64,
  seq_len 1024). Other shapes go through the generic code paths, which were checked only by forcing
  the generic attention path on this model.
