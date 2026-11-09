# MJ — llama2.c in Mojo 1.2.0

Single source file `mj.mojo` → executable `mj`. It implements `reference/run.c` (forward pass, BPE
tokenizer, greedy argmax) behind the line protocol in `INTERFACE.md`. It uses only the Mojo 1.2.0
standard library: SIMD, compile-time parameters, `std.math`, file I/O and raw fd reads. It does not
use Python interop, MAX, `parallelize`, threads, fast-math or any external math library. The
program itself creates no threads; see Limitations for the idle Mojo runtime pool.

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

Speed (Apple M1, one thread, after the profiling round below):
- about 133 positions/s on the development workload;
- about 121 tok/s averaged over a full 1024-position `FIXED`;
- 0.09 s to `READY`.

The forward pass takes 7.42 ms/token. Reading the weights alone costs about 7.0 ms/token at the
measured 62.5 GB/s single-core streaming limit, so the program runs at about 95% of the memory
bound.

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

- **matmul:** uses a packed weight layout; see "Profiling round" below. When the input length
  `n` is not a multiple of 8, it falls back to row-major blocks of 2 rows × 32 lanes. Tails handle
  any `n` and `d`.
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
  `READY`. Packing also needs a temporary buffer the size of the largest matrix while loading
  (98 MB for the classifier); it is freed before `READY`.
- The Mojo runtime starts its own worker pool of 4 threads at process start. I do not create or
  configure them. `sample` shows them 100% blocked in `semaphore_wait_trap` for the whole run; all
  compute runs on the main thread. I left them alone because the rules forbid setting thread counts.
- The public check only exercises this model (dim 768, 12 layers, 12 heads, head size 64,
  seq_len 1024). Other shapes go through the generic code paths, which were checked only by forcing
  the generic attention path on this model.

## Profiling round (performance pass 2)

The tools are in `prof/`:
- `work.py`: development workload driver;
- `instrument.py`: generates a timed copy of `mj.mojo` (`mj.mojo` itself has no profiling code);
- `startup.py`: time to `READY`;
- `mb.mojo`, `mb2.mojo`, `bw2.mojo`: microbenchmarks.

**Workload.** Only the supplied development inputs were used. `prof/work.py` replays the three
`fixtures.json` prompts the way `check.py` drives them: `GENERATE 64 <text>` (checked against the
fixture ids), then `LOGITS` over the fixture `seq`. That is 230 positions per pass, about 75% of
them prompt or forced positions, all at contexts of at most 82.

```
mojo build -g1 mj.mojo -o /tmp/mjprof/mj_g             # line tables for symbolised samples
python3 prof/work.py /tmp/mjprof/mj_g . 4 & /usr/bin/sample <pid> 10 -mayDie
python3 prof/instrument.py mj.mojo /tmp/mjprof/mj_prof.mojo && mojo build /tmp/mjprof/mj_prof.mojo -o /tmp/mjprof/mj_prof
python3 prof/work.py /tmp/mjprof/mj_prof . 2           # per-phase ms/token on stderr at QUIT
```

### Evidence

**1. `sample`, 10 s on the start-of-pass build.** Main thread: `matmul` 7124 of 7293 samples
(98%), the rest of `forward` (RoPE, attention, norms) 106 (1.5%), `softmax` 5, `write` 29. The four
other threads are the Mojo runtime pool, 7293/7293 samples in `semaphore_wait_trap`.

**2. Per-phase timers.** These are ms/token averaged over 914 forwards. GB/s is the bytes of weights
streamed divided by the time.

| phase | before | after packing |
|---|---|---|
| q,k,v (84.9 MB) | 1.52 ms (55.8 GB/s) | 1.43 ms (59.3) |
| wo (28.3 MB) | 0.50 ms (56.2) | 0.47 ms (59.9) |
| w1+w3 (151 MB) | 2.66 ms (56.8) | 2.51 ms (60.2) |
| w2 (75.5 MB) | **1.67 ms (45.3)** | 1.25 ms (60.3) |
| classifier (98.3 MB) | 1.74 ms (56.6) | 1.63 ms (60.2) |
| RoPE + attention | 0.08 ms | 0.08 ms |
| forward total | 8.20 ms | 7.42 ms |

**3. Bandwidth ceiling (`bw2.mojo`).** Summing the 438 MB buffer gives 62.5 GB/s with 1 sequential
stream, 61.6 with 2, 57.4 with 4 and 58.7 with 8 far-apart streams. One sequential stream is
best, and the floor is about 7.0 ms/token.

**4. Matvec microbenchmark (`mb.mojo`).** It runs over all the model's matrix shapes back to back:

| kernel | GB/s |
|---|---|
| row-major, 2 rows × 32 lanes (the old kernel) | 59.1 |
| row-major, 4 rows × 16 lanes | 41.9 (multiple strided streams hurt) |
| packed, R=2..16 rows, W=8..32 lanes | 62.0–62.8 (at the ceiling) |

`mb2.mojo` shows that ordering the matrices contiguously in execution order rather than as in the
checkpoint makes no difference (62.5–63.1 GB/s either way), so I did not re-layout across matrices.

### Change made: packed weight layout

At load time each matmul weight (wq, wk, wv, wo, w1, w2, w3 per layer, and the classifier) is
rearranged into blocks of 8 rows. Within a block the rows are interleaved in 8-float chunks:
`[r0 c0..7][r1 c0..7]..[r7 c0..7][r0 c8..15]..`.

The matvec then reads one strictly sequential stream and keeps 8 independent 8-lane FMA
accumulators. Each row's dot product is still a SIMD-lane FMA accumulation followed by a reduction,
so the arithmetic per row is the same kind as before. Only the lane width changed (32 → 8), which
moves rounding at the 1e-7 relative level.
- Rows past the last full block stay row-major.
- Matrices whose `n` is not a multiple of 8 stay row-major and use the old kernel.
- The values, float32 precision, shapes (one matvec per weight per token) and work are unchanged.
  This is a memory-layout change only; no positions are batched.

**Shared classifier.** When the classifier is shared, the token embedding table is the packed
matrix. The embedding lookup reads its row back out of the packed layout (`packed_row`). When the
classifier is unshared, the embedding table stays row-major.

**Loading.** Weights are now read section by section in file order. Each matrix goes from the file
into a temp buffer and is packed into its final slot, which avoids an extra full copy. Packing
initially cost 55 ms of startup (0.078 → 0.134 s); this version costs about 20 ms (0.069 → 0.090 s
in the same session). That equals about 3 tokens of forward time.

**Block shape.** In the full program, R×W = 8×8, 4×16, 8×16, 16×8 and 4×8 all measured 134–136 tok/s
(FIXED 128, three interleaved rounds), which is within noise. I kept 8×8 because it uses 16
accumulator registers and the fewest `x` reloads per weight byte.

### Result (interleaved A/B, same session, `prof/work.py … 2`, three rounds)

| | before (start-of-pass source) | after |
|---|---|---|
| GENERATE workload | 124.0 / 123.8 / 124.1 pos/s | 135.3 / 133.4 / 128.7 pos/s |
| LOGITS workload | 120.4 / 116.2 / 123.0 pos/s | 133.0 / 134.7 / 126.0 pos/s |
| FIXED 1024 | 111.3 tok/s | 121.4 tok/s |
| time to READY | 0.069 s | 0.090 s |

That is about 8–10% faster per position.

### Considered and rejected

- **Software prefetch** (pass 1) and multiple concurrent streams (evidence 3): no gain or slower.
- **More attention work:** it is 1% of the time on these workloads.
- **Processing several prompt or FIXED positions per weight pass (prefill batching):** this would
  make the forward compute-bound and much faster. I did not do it, because the rules require
  batch size 1 and the same operation shapes (matrix-vector per token).

### Tests after the change

- `python3 check.py -- ./mj`: PASS.
- `python3 check.py --tokens 64 --prompts 3 -- ./mj`: PASS, max logit error 1.2e-5 to 2.9e-5.
- My scripts (`/tmp/mjtest`), all as before:
  - all fixture `PIECE`s match;
  - `FIXED` argmaxes reproduce the fixture continuations;
  - pipelined input works;
  - `GENERATE 0`, `stop` and `context` (148 tokens after an 877-token prompt) all end correctly;
  - long-run outputs are unchanged (the `GENERATE 2000` run still stops at 238 tokens; same
    `FIXED 1024` argmaxes).
- **Unshared-classifier path:** I built a temporary checkpoint with `-vocab_size` and an appended
  `wcls` equal to the embedding table. `LOGITS` were bit-identical to the shared model's, and
  `GENERATE` output was identical. The file was deleted afterwards because the disk is 98% full.
- **Non-default code paths, checked with temporary builds:**
  - forcing the row-major fallback (`is_packed` → False): full check PASS;
  - `PK_R=5`, which leaves 3 row-major tail rows per matrix and puts embedding rows in the tail
    and in packed blocks: full check PASS.

### Remaining limitations

- The forward pass is at about 95% of the single-core memory-bandwidth floor. Further speedup
  would need fewer bytes per token, which means quantization (forbidden), or batching positions
  (rejected above).
- The profile covers only the short development contexts (at most 82 positions). Long contexts
  (FIXED 1024) were timed but not profiled phase by phase.
