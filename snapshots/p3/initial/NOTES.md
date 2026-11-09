# P3 — NumPy float32 port of reference/run.c

Program: `python3 run.py <checkpoint.bin> <tokenizer.bin>` (also executable as `./run.py`).

## Commands run
```
cat RULES.md INTERFACE.md check.py; read reference/run.c
python3 -c ...            # inspected fixtures.json, model header (768,2048,12,12,12,32000,1024), tokenizer.bin
                          #   (27 max token length, no NUL bytes, no duplicate pieces, 256 <0xNN> byte pieces)
chmod +x run.py
python3 check.py -- python3 run.py                          # PASS (max logit error ~1.9e-5)
python3 check.py --tokens 64 --prompts 3 -- python3 run.py  # PASS (max logit error <= 2.5e-5, tol 5e-4)
printf 'PIECE ...\nFIXED ...\nGENERATE 0 hi\nGENERATE 5 \nENCODE\nQUIT\n' | python3 run.py model.bin tokenizer.bin
printf 'GENERATE 2000 Once upon a time\nQUIT\n' | python3 run.py ...   # DONE 238 stop
python3 - (1021-token prompt, GENERATE 50)                  # DONE 4 context (= 1024 - 1021 + 1)
```

## What was implemented (run.py)
- Checkpoint: 7-int header; negative vocab_size => unshared classifier; weights laid out exactly as
  `memory_map_weights` (including skipping the legacy freq_cis tables). Loaded once with `np.fromfile` as float32.
- Forward pass per token, per position, batch 1, all layers, float32 throughout: rmsnorm (1e-5, `w*(ss*x)`),
  Q/K/V/O/W1/W2/W3/classifier as separate float32 matrix-vector products (`np.dot` -> Accelerate sgemv),
  RoPE with the reference formula (`1/powf(10000, hd/hs)`, `pos*freq`, cos/sin in float32, precomputed table),
  attention over positions 0..pos with `score / sqrtf(head_size)`, max-subtracted softmax, GQA-aware
  (kv_mul) head grouping, SwiGLU `x*(1/(1+exp(-x)))*w3x`, residuals, final rmsnorm, full logits every step.
- KV cache: float32, (layer, kv_head, seq_len, head_size) so each head slice is contiguous. Each command
  resets it (positions written by the previous command are zeroed) and starts at position 0.
- Tokenizer: byte-exact port of `encode` (dummy-prefix " ", UTF-8 codepoint grouping capped at 4 bytes,
  byte fallback +3, greedy highest-score merge with first-index tie break and the same -1e10 start) and
  `decode` (strip leading space after BOS, `<0xNN>` -> raw byte; byte 0 decodes to an empty C string).
- Protocol: stdin read as raw bytes; only the trailing `\n` is removed so leading/trailing/multiple spaces
  and tabs survive. GENERATE forces prompt tokens, then greedy `np.argmax` (lowest index wins ties);
  `stop` on token 1 (not printed), `max` after max_new tokens (checked first), `context` when seq_len
  positions have been forwarded (the token from position seq_len-1 is still emitted, as in run.c).
- No threads started, no thread counts set, no fast-math, no float64 in the model computation.

## Limitations / judgement calls
- `PIECE` for a piece that decodes to nothing (e.g. `<0x00>`, or " " after BOS) prints `HEX ` with an
  empty hex string (trailing space). INTERFACE.md does not pin this down.
- Summation order differs from the C scalar loops (BLAS sgemv/sdot, NumPy pairwise sums), so logits
  differ from the reference by ~1e-5 (well inside the 5e-4 tolerance); argmax near-ties could in principle flip.
- Argmax with NaN logits would follow NumPy (first NaN) rather than C's comparison semantics; not expected.
- FIXED/LOGITS with more ids than seq_len is not supported (the reference would overrun its cache too).
- Rough speed on this machine: ~13 ms/token including Python overhead (238 generated tokens + load in ~3.6 s).
- Harness, fixtures, rules and reference were not modified. Only check.py's public inputs were used.
