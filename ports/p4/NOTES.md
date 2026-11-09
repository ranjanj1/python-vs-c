# P4 — Numba port of reference/run.c

Program: `python3 run.py <checkpoint.bin> <tokenizer.bin>` (protocol per INTERFACE.md).
Check: `python3 check.py -- python3 run.py`.

## Result (current version, after the second performance pass)

- `python3 check.py -- python3 run.py` → **PASS** (max logit error 9.8e-6, tolerance 5e-4).
- `python3 check.py --tokens 64 --prompts 3 -- python3 run.py` → **PASS** (all 3 prompts, 64 tokens,
  max logit error 2.4e-5). These errors are unchanged from the first version: both optimization
  passes give logits **bitwise identical** to the first version.
- Dev workloads (median of 3 sessions, `scratch/profile_pass3.txt`): GENERATE 64 runs at
  95–114 tok/s, LOGITS at 117–120 positions/s, and a full check-equivalent session takes
  4.7–4.9 s (first version: 7.5 s). Startup to `READY` takes ~1.0 s with a warm Numba cache.
  The peak memory footprint is 534 MB (previous pass: 705 MB).

## What was built (`run.py`)

- **Model loading**: reads the 7-int header; a negative vocab_size means unshared classifier
  weights. Memory-maps the float32 weights in run.c order and skips the two legacy freq_cis
  blocks. Each weight matrix is copied once, by a compiled kernel (`tile_into`), into a
  **column-tiled transposed layout** `(tiles, in, 128)`. Tile `t` holds output rows
  `128t … 128t+127` of `W`, stored input-major. Values are unchanged float32 and only the layout
  differs. The q/k/v tiles share one array per layer, and so do the w1/w3 tiles. The classifier
  gets its own tiled copy (it is shared with the embedding in this checkpoint). With shared
  weights there is no separate embedding copy: `embed()` reads row `token` out of the tiled
  classifier (the same float32 values). An unshared embedding gets its own tiled copy. A dimension
  that isn't a multiple of 128 is zero-padded to whole tiles. The padded outputs are never read.
- **Forward pass** (`forward`, all `@njit` nopython, `cache=True`, no `fastmath`, no `parallel`):
  - same ops as run.c, in the same order: rmsnorm → q/k/v → RoPE → attention with softmax over
    0..pos → wo → residual → rmsnorm → w1/w3 → SwiGLU → w2 → residual → final rmsnorm →
    classifier. The classifier runs at every position, forced prompt positions included, as in
    run.c.
  - **matvec** `matvec_tiles`/`matvec_tile`: for each tile, `out[c] += Wt[j,c]*x[j]` with the loop
    over the 128 outputs on the inside, 4 input rows per pass. This vectorizes without fastmath
    because no reduction is reordered. Each output is a float32 sum over `j` in ascending order,
    exactly as run.c's `matmul` computes it. A test (`scratch/test_tiling.py`) shows the result is
    bitwise equal to a scalar sequential float32 sum. No BLAS, no `np.dot`, no `@`.
  - **KV cache**: per layer, keys are stored as `(n_kv_heads, seq_len/64, head_size, 64)`, i.e.
    transposed within blocks of 64 positions. The q·k scores vectorize over the block's `t`, and
    each score is still summed over `i` in order. Values are stored `(seq_len, kv_dim)`. Each
    score is divided by `sqrtf(head_size)` as in run.c. Every command starts with an all-zero
    cache at position 0. `reset()` clears only the positions written since the previous reset;
    every other entry is still zero.
  - **RoPE**: same expressions as run.c (`powf`, `cosf`/`sinf` of `pos*freq`). The cos/sin pairs
    depend only on (i, pos), so they are computed once per position and reused by all layers
    (run.c recomputes them per layer). The values are bit-identical; this saves ~0.06 ms/token.
  - Checked with `inspect_types` and the LLVM IR for every kernel: no float64 variables, no
    double arithmetic, no fast-math flags. The only float64 values are literal constants
    immediately cast with `np.float32(...)`.
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

## Performance pass: profiling evidence

Machine: Apple M1 (8 GB RAM), shared with other load (load average 2–3.5, heavy memory
compression), so every number is a median or best of repeated, interleaved runs.
Workloads: the supplied development workloads only. These are the `fixtures.json` prompts as
`check.py --tokens 64 --prompts 3` drives them: 9 ENCODEs, then GENERATE 64 and LOGITS over the
full 69/79/82-token sequences for each prompt. No other inputs were used.

Tools: `scratch/profile.py` (`e2e` drives `run.py` over the protocol like check.py and times each
command; `phases` times each phase of a forward pass on prompt 3's 82-position sequence using the
same compiled kernels), plus the microbenchmarks listed under "Commands run".

**Before** (first version; `scratch/profile_before.txt`): 10.0 ms/token, 44.8 GB/s effective.

| phase | ms/token | share | GB/s |
|---|---|---|---|
| classifier (768→32000) | 3.04 | 30.1% | 32.3 |
| w1, w3, w2 | 1.47 + 1.46 + 1.39 | 42.6% | 51–55 |
| q, k, v, o | 4 × 0.53 | 21.0% | 53 |
| attention | 0.44 | 4.4% | |
| rope / swiglu / rmsnorm / residual | 0.07 / 0.06 / 0.04 / 0.02 | 1.8% | |

Findings:
- The run is memory-bandwidth bound: 438 MB of weights are streamed per token. A pure streaming
  loop in Numba measured ~49–56 GB/s on one core (`scratch/bench_bw.py`).
- The classifier was the outlier at 32 GB/s. Its 128 KB output vector is reloaded and stored for
  every few input rows, so traffic spills out of L1.
- The layer matvecs ran at 51–55 GB/s. Column tiles of 128 outputs reached ~60–62 GB/s
  (`bench_tile.py`, `bench_tile2.py`): each tile is one contiguous stream, and its 512-byte
  output slice stays in L1. Inside a tile, a 4-row unroll was best (u1 ~49, u2 ~60,
  u4 ~62 GB/s).
- **Rejected:** fusing q/k/v or w1/w3 into one *untiled* wide matrix was slower (50 vs 54 GB/s;
  40 vs 51 GB/s), because wider outputs stream worse (`bench_fuse.py`). Tile widths 64/256 and
  an 8-row unroll were within noise of T=128 with 4 rows (`bench_tile3.py`). With a 2-row
  kernel, classifier tiles of 256–1280 columns reached ~1.9 ms (min), and wider tiles got worse
  (3200 columns: 2.5 ms) (`bench_cls*.py`).

**After** (`scratch/profile_after.txt`): ~7.7–8.2 ms/token, 55–57 GB/s effective.

| phase | ms/token | share | GB/s |
|---|---|---|---|
| w1\|w3 (tiled, one call) | 2.57 | 32.6% | 58.8 |
| classifier (tiled) | 1.65 | 21.0% | 59.5 |
| q\|k\|v (tiled, one call) | 1.44 | 18.3% | 58.9 |
| w2 | 1.28 | 16.2% | 59.0 |
| o | 0.49 | 6.2% | 57.9 |
| attention | 0.25 | 3.1% | |
| everything else | 0.21 | 2.6% | |

A/B over the three dev sequences (`scratch/ab.py`, old = `scratch/run_v1.py`, interleaved):
logits bitwise identical on all 69+79+82 positions. Forward time went from 8.92 to 7.71 ms/token
(1.16×) in one run and from 9.56 to 8.32 (1.15×) in another.

End-to-end on the dev workloads (`profile.py e2e`, medians; before → after):

| metric | before | after |
|---|---|---|
| GENERATE 64, prompt 0 / 1 / 2 (s) | 0.71 / 0.79 / 0.94 | 0.56 / 0.67 / 0.68 |
| LOGITS 69 / 79 / 82 positions (s) | 0.74 / 1.01 / 0.97 | 0.62 / 0.66 / 0.69 |
| whole check-equivalent session (s) | 7.54 | 5.41 |
| startup to READY (s) | 1.70 | 1.45 |

The 1016–1121-token context tests in `scratch/edge.py` went from ~15.8 s to ~10.8–11.2 s.

## Performance pass: changes

1. **Column-tiled weights** (T = 128) for every matvec, including the classifier. Same
   arithmetic and order per output; this alone gives the bulk of the gain (classifier
   3.0 → 1.65 ms, layers +8–10% bandwidth).
2. **q|k|v and w1|w3 tiles share one array** per layer, so each group is a single
   `matvec_tiles` call. With tiling this is just a longer tile list, so every output is still its
   own sum.
3. **Attention**: two key rows per pass for the scores and two timesteps per pass for the value
   sum. The adds stay in the same order, so results are bitwise identical (0.44 → 0.25 ms).
4. **RoPE cos/sin computed once per position** instead of per layer (bit-identical values).
5. **Loading**: the checkpoint is memory-mapped instead of read into a fresh 438 MB array, and the
   tiling copy uses branch-free 16×16 blocks. In fresh processes on this memory-pressured machine,
   `Model()` dropped from ~0.5–0.8 s (read + tile) to ~0.37–0.40 s.

Unchanged: tokenizer, protocol handling, KV-cache semantics, float32 everywhere, one thread.

## Performance pass: tests

- `python3 check.py -- python3 run.py` and `... --tokens 64 --prompts 3 ...`: PASS, with the same
  errors as before.
- `scratch/ab.py`: new forward vs the first version, bitwise identical on all three dev
  sequences.
- `scratch/test_synthetic.py`: synthetic checkpoints covering the paths the dev model never
  exercises. These are dimensions that aren't multiples of 128 (padding), GQA (`n_kv_heads` 1 or 4
  < `n_heads`), unshared classifier weights (negative vocab) and running up to seq_len. They are
  bitwise identical to the first version at every position.
- `scratch/test_tiling.py`: `tile_into` round-trips exactly with zero padding, and
  `matvec_tiles` is bitwise equal to a scalar sequential float32 sum in run.c order.
- `scratch/edge.py`: PIECE vs all fixture pieces, empty prompt, `GENERATE 0`, FIXED argmax
  consistency, and `context` stop at seq_len (9 tokens after a 1016-token prompt): all as before.
- IR/type inspection of every kernel (`NUMBA_CACHE_DIR=$(mktemp -d)`): no float64, no double
  arithmetic, no fast-math flags.

## Second performance pass: profiling evidence

Same machine and conditions (M1, 8 GB, load average 2.4–6.3, heavy memory compression). Same
supplied dev workloads only (the `fixtures.json` prompts as check.py drives them).
Saved output: `scratch/profile_pass3.txt`. For A/B, `scratch/run_v2.py` is the previous pass's
version, rebuilt by reverting this pass's edits. It reproduces that version's logits bitwise.

What the profile showed:
- **Forward pass** (`profile.py phases`): 91% matrix-vector work at the bandwidth plateau. All
  tile-width and unroll variants measured within noise last pass, and every other phase is
  under 3% at ≤82 positions. No large lever remains there.
- **Outside the forward pass** (`scratch/overhead.py`): dispatch of the 29-argument `forward()`
  is 1.1 µs, argmax 40 µs per token, the full KV reset 1.3 ms per command, and the LOGITS write
  10.3 ms per command (7.6 ms without the redundant `astype` copy). Each is ≤1.5% of its
  command.
- **Startup** (`scratch/startup_profile.py`, fresh processes): `import numpy` + `import numba`
  0.6–0.85 s, model load 0.45–1.6 s, first forward (cached-kernel load) 70–130 ms.
  `scratch/load_parts.py` separated the load cost. When the OS had evicted the checkpoint from
  the file cache, the first touch of its pages cost 952 ms. Otherwise it was ~25 ms, and the
  load itself took 130–200 ms. The only lever here is our memory footprint.
- **Attention / key store** (`scratch/bench_kv.py`, 12 layers, caches flushed as the weight
  stream does, median of 9): one position's key was written as 64 values 4 KB apart. A
  blocked layout of 64 positions was 1.17× faster at 40 positions, 1.21× at 80, 1.59× at 300 and
  1.01× at 1000. Block 32 was slower at 1000 (0.95×), and block 128 was similar to 64.
  Softmax is only ~11% of attention even at 1000 positions (`bench_attn_parts.py`).
- **Rejected:** one flat weight buffer in consumption order (`scratch/bench_flat2.py`, one
  process per layout) was within noise (minimums 8.29–8.37 vs 8.39 ms/token). A strided-view
  version of the same idea (`bench_flat.py`) was 2.5× slower, because Numba does not vectorize
  non-contiguous arrays.

## Second performance pass: changes

1. **Blocked key cache** (64 positions per block, see above). Bitwise identical results.
2. **No row-major embedding copy with shared weights**: the token lookup reads the tiled
   classifier.
3. **Partial KV reset**: only positions written since the last reset are zeroed. The cache is
   all-zero at the start of every command, as before, and never-used cache pages are not touched.
4. **LOGITS write**: the array is allocated as little-endian float32 and written directly,
   without the extra converted copy.

Results, previous pass's version → current, alternating processes:

| measurement | previous | current |
|---|---|---|
| forward, dev sequences (`ab.py 8 run_v2`) | 8.29 ms/token | 8.25 ms/token (1.004×, noise) |
| whole dev session (`profile.py e2e`, 2 rounds) | 4.70 / 5.39 s | 4.70 / 4.93 s |
| startup to READY (same runs) | 0.99 / 1.29 s | 0.98 / 1.00 s |
| FIXED 300 positions (`longctx.py`, 3 runs) | 2.55–2.61 s | 2.47–2.48 s |
| FIXED 1024 positions (3 runs) | 10.39–10.41 s | 9.91–9.99 s |
| peak memory footprint (`/usr/bin/time -l`) | 705 MB | 534 MB |

On the dev workloads this pass is a small gain: the session time is within noise to ~9%, mostly
from startup. It is about 5% at 300–1024 positions, and memory is 24% lower, which matters on
this 8 GB machine where the checkpoint was seen being evicted from the file cache.

## Second performance pass: tests

- `check.py` quick and `--tokens 64 --prompts 3`: PASS, same errors as before.
- `scratch/ab.py 8 run_v2` and `ab.py 6` (vs the first version): logits bitwise identical on all
  three dev sequences.
- `scratch/longctx.py`: FIXED 300 and FIXED 1024 answers identical (SHA-1) between the previous
  and current versions.
- `scratch/test_synthetic.py`: padding, GQA (1 and 4 kv heads), unshared classifier/embedding,
  sequence lengths 40 and 33 (partial key blocks) — all bitwise identical to the first version.
- `scratch/test_reset.py`: after a 161-position command and a reset, the cache is all-zero, and a
  20-position sequence gives bitwise the same logits as on a fresh model.
- `scratch/edge.py`: unchanged behaviour, including the `context` stop at seq_len.
- `scratch/test_tiling.py`, plus IR/type inspection of every kernel including `embed`: no float64,
  no double arithmetic, no fast-math flags.

## Considered and not done: multi-position prefill

The largest remaining speedup on the dev workloads would come from processing a command's
known tokens (the forced prompt tokens in GENERATE, and every token of FIXED/LOGITS) several
positions at a time. Each weight tile would then be read once for many positions instead of once
per position. The arithmetic per output could keep the same order and give the same bits, but
that turns the reference's per-token matrix-vector products into matrix-matrix products. The
rules say "Batch size 1" and, for CB, "same operation shapes", and INTERFACE.md describes FIXED
as "full logits each step". I read that as excluding it and did not implement it. It would need an
explicit ruling.

## Commands run

First implementation:
```
cat RULES.md INTERFACE.md check.py reference/run.c; head fixtures.json
python3 -c "import numba, numpy; print(numba.__version__, numpy.__version__)"   # 0.68.1 2.5.4
python3 -c "<read model header>"            # (768, 2048, 12, 12, 12, 32000, 1024)
python3 -c "<scan tokenizer.bin>"           # no duplicate pieces, 256 well-formed <0xHH> tokens
python3 scratch/bench_mm.py                 # dot-product kernels: lane-split versions don't vectorize w/o fastmath
python3 scratch/bench_mm2.py                # transposed axpy form ~3x faster than best dot form
python3 scratch/bench_mm3.py                # unroll/tiling variants (strided tiling was slower)
cd scratch && PYTHONPATH=.. python3 bench_unroll.py   # 2-row unroll best for layer mats, 4-row for classifier
python3 check.py -- python3 run.py
python3 check.py --tokens 64 --prompts 3 -- python3 run.py
NUMBA_CACHE_DIR=$(mktemp -d) python3 - <<EOF ... inspect_types / inspect_llvm ... EOF   # float32-only, no fast flags
python3 scratch/speed.py                    # startup + tok/s
python3 scratch/edge.py                     # PIECE vs fixtures, empty prompt, max_new 0, FIXED, context stop
```

Performance pass:
```
/usr/sbin/sysctl -n machdep.cpu.brand_string hw.memsize ...; uptime; vm_stat   # M1, 8 GB, loaded
python3 scratch/profile.py all 3 | tee scratch/profile_before.txt     # baseline evidence
python3 scratch/bench_cls.py; python3 scratch/bench_cls2.py         # classifier tile layouts
python3 scratch/bench_bw.py                                          # single-core streaming ceiling
python3 scratch/proto.py 6                                           # first prototype (fused untiled): bitwise ok, no gain
python3 scratch/bench_fuse.py                                        # fusion untiled slower; tiled classifier faster
python3 scratch/bench_tile.py; python3 scratch/bench_tile2.py        # tile width / unroll for all shapes
cp run.py scratch/run_v1.py                                          # keep first version for A/B
python3 scratch/ab.py 8                                              # bitwise + timing A/B
python3 scratch/bench_tile3.py                                       # T=64/128/256, u4/u8: plateau
python3 scratch/bench_load.py; python3 scratch/bench_modelload.py    # load-time copy kernels
python3 scratch/fresh_load.py {fromfile|memmap}                      # fresh-process load time
python3 scratch/test_tiling.py; python3 scratch/test_synthetic.py    # padding/GQA/unshared tests
python3 check.py -- python3 run.py; python3 check.py --tokens 64 --prompts 3 -- python3 run.py
python3 scratch/edge.py
NUMBA_CACHE_DIR=$(mktemp -d) python3 - <<EOF ... inspect_types / inspect_llvm ... EOF
python3 scratch/profile.py all 4 | tee scratch/profile_after.txt      # after evidence
```

Second performance pass:
```
python3 scratch/startup_profile.py           # x3: fresh-process startup breakdown
python3 scratch/load_parts.py                # x3: file-page vs anonymous-page faults vs copy
python3 scratch/overhead.py                  # dispatch, argmax, reset, LOGITS write costs
python3 scratch/bench_flat.py; for i in 1 2 3; do python3 scratch/bench_flat2.py sep; python3 scratch/bench_flat2.py flat; done
python3 scratch/bench_kv.py [32|64|128]      # blocked key-cache layouts, bitwise + timing
python3 scratch/bench_attn_parts.py          # softmax share at long context
python3 check.py --tokens 64 --prompts 3 -- python3 run.py
python3 scratch/test_synthetic.py; python3 scratch/test_reset.py; python3 scratch/test_tiling.py
cp run.py scratch/run_v3.py                  # snapshot of this pass's version
python3 - <<EOF ... revert this pass's edits ... EOF  -> scratch/run_v2.py   # previous version for A/B
python3 scratch/ab.py 8 run_v2; python3 scratch/ab.py 6
python3 scratch/longctx.py scratch/run_v2.py; python3 scratch/longctx.py run.py   # x3, alternating
PROG=scratch/run_v2.py python3 scratch/profile.py e2e 3; python3 scratch/profile.py e2e 3   # x2, alternating
printf 'QUIT\n' | /usr/bin/time -l python3 {scratch/run_v2.py|run.py} model.bin tokenizer.bin   # peak memory
python3 check.py -- python3 run.py; python3 scratch/edge.py
NUMBA_CACHE_DIR=$(mktemp -d) python3 - <<EOF ... inspect_types / inspect_llvm ... EOF
python3 scratch/profile.py phases 4; python3 scratch/profile.py e2e 3   # -> scratch/profile_pass3.txt
```

`scratch/` holds only benchmark/test helpers, the saved profiles, and earlier versions for A/B:
`run_v1.py` (first version), `run_v2.py` (after the first performance pass) and `run_v3.py`
(a snapshot of the current `run.py`). `run.py` does not use any of it.

## Limitations / choices to note

- **Bandwidth ceiling**: 91% of the time is now matrix-vector work at ~58–60 GB/s, close to what
  one M1 core streamed in every variant tried. The remaining big wins would need fewer weight
  bytes (quantization) or more cores, and the rules forbid both. Further tuning attempts (tile
  width, unroll depth) were within noise.
- **Summation order**: each matvec output is summed in the same order as run.c. Numba emits no
  FMA (no `contract` flag), while the build that produced the fixtures may have fused
  multiply-adds. The differences are ~1e-5 in the logits.
- **Memory**: the tiled weights (438 MB) and the tiled classifier (98 MB, which also serves the
  embedding lookup when shared) stay resident. The peak footprint is 534 MB. The checkpoint is
  mapped read-only during load. Load time depends heavily on OS memory pressure: when the
  checkpoint has been evicted from the file cache, reading it costs ~1 s more.
- **Long contexts**: attention grows with position. It is ~3% of a token at the dev workloads'
  ≤82 positions, but ~4.2 ms per token (about half) near position 1000: score and value
  arithmetic plus KV reads, with softmax ~11%. The blocked key cache helped up to ~300
  positions. Beyond that, vectorizing `expf` would change its results, and multi-position
  prefill is excluded (see above), so it was not pursued further.
- **Startup** (~1.0 s, more under memory pressure): ~0.6–0.85 s is importing numpy+numba (not
  reducible from here), ~0.15–0.2 s is loading and tiling, and ~0.1 s is the warm-up forward,
  which loads the cached kernels.
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
- **Measurement noise**: the machine was shared and memory-pressured throughout. Identical
  commands varied by ±15–30% between runs, so treat the figures as medians under that load.
