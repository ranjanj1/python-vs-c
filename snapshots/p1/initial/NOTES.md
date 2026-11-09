# P1 — pure Python port of reference/run.c

Implementation: `run.py` (Python 3.12 standard library only: `sys`, `math`, `struct`, `array`).
Start: `python3 run.py <checkpoint.bin> <tokenizer.bin>`, then the INTERFACE.md line protocol.

## Commands run

```
python3 --version                      # Python 3.12.13 (/Users/Shared/pvc-kit/stdlib/bin/python3)
python3 -c '...struct.unpack("7i")...' # read model header: dim 768, hidden 2048, 12 layers,
                                       # 12 heads, 12 kv heads, vocab 32000 (shared cls), seq_len 1024
python3 check.py                       # only to learn the invocation (needs `-- <program>`)
python3 -I /tmp/p1bench/b.py           # micro-benchmark of pure-Python dot-product forms
printf 'ENCODE ...\nPIECE ...\nFIXED ...\nGENERATE ...\nQUIT\n' | python3 -I run.py model.bin tokenizer.bin
python3 check.py -- python3 -I run.py model.bin tokenizer.bin
```

I did not read check.py, fixtures.json or fixtures.f32 (RULES.md 6). I read only
RULES.md, INTERFACE.md and reference/run.c.

## What was implemented

- Checkpoint loading exactly as `read_checkpoint`/`memory_map_weights`: 7-int header,
  negative vocab size means an unshared classifier, and the freq_cis block is skipped.
  Weights are kept as one float32 `array('f')`, which is read into memory rather than mmapped.
- `forward()` follows run.c op for op: rmsnorm, then the q/k/v matmuls, RoPE (per-position cos/sin of
  float32 `pos*freq`), and the KV cache write at `pos`. Next comes multi-head attention over 0..pos
  with GQA indexing (`h / kv_mul`), the softmax, the weighted value sum and the wo matmul with
  its residual. Then the FFN: rmsnorm, w1/w3, SwiGLU, w2 and its residual. Finally the last
  rmsnorm and the full classifier at every position (the prompt steps in GENERATE too).
- KV cache: float32 `(layer, seq_len, kv_dim)` arrays, zeroed at the start of every command.
- Tokenizer: same file format, the same `encode` (dummy prefix, UTF-8 codepoint grouping,
  byte fallback +3, greedy highest-score pair merging with first-index tie-break) and the same
  `decode` (BOS leading-space strip, `<0xHH>` raw bytes, as `sscanf` parses them). A dict
  replaces the bsearch over the sorted vocab. Both do an exact string match.
- Protocol: ENCODE, PIECE, GENERATE (greedy argmax with the lowest index winning ties, prompt
  tokens forced, `stop` on token 1, `max`, `context` at seq_len), FIXED, LOGITS (little-endian
  float32, n x vocab), QUIT. stdout is flushed after every line. One thread. No timing output.

## Precision (limitation)

The Python standard library has no float32 scalar arithmetic. All stored tensors are float32:
weights, activations after each op, q/k/v, the KV cache, attention weights and logits.
Scalar steps in rmsnorm, RoPE, softmax and SwiGLU are rounded to float32 through `struct`,
in the reference's order. **The running sums of the dot products (matmuls, attention scores
and the weighted value sum) are Python floats (binary64), rounded to float32 when stored.**
They accumulate left to right in the same order as the C loops.
Rounding every partial sum to float32 would need a struct round-trip per multiply-add, about
40x slower. That makes this ~110M-parameter model unusable, so I did not do it. As a result
the logits differ from the C reference by small float32-level rounding differences, not bit-exactly.
`math.exp/cos/sin/sqrt/pow` are double-precision libm calls rounded to float32. `expf` etc.
may differ by 1 ulp.

## Other choices and limitations

- "Scalar loops": the matmul inner loop is an explicit `for a, b in zip(row, x): val += a*b`.
  I did not use `math.sumprod`, `sum(map(mul, …))`-style kernels or any vectorised helper.
  Measured cost is ~26 ns per multiply-add, roughly 2.6 s per token at short context. Long
  GENERATE/FIXED runs are therefore very slow (minutes per 100 tokens).
- PIECE for the `<0x00>` byte token prints `HEX 00` (the decoded byte). C's `safe_printf`
  would treat that piece as an empty string.
- Inputs longer than seq_len: FIXED/LOGITS process only the first seq_len ids. GENERATE reports
  `context`. `GENERATE 0 …` answers `DONE 0 max` without running the model.

## Validation result

(see below)
