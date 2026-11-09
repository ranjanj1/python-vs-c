# Rules (draft, frozen after harness pilots; see bench/protocol.json for the hash)

Goal: implement one version of the supplied llama2.c inference program, fast and correct, under these rules.

1. Same work. Same model, tokenizer, architecture, float32 weights, positions, KV-cache behaviour and stop policy as the reference.
   Batch size 1. No quantization, skipped layers, reduced work, prompt-specific shortcuts or cached answers.
2. Follow bench/INTERFACE.md exactly. The harness times everything from outside.
3. One thread of compute. Do not start threads or set thread counts. The harness sets them.
4. Use only your allowed techniques (below) and the supplied bundle. No web, no downloads, no other ports.
5. No fast-math in code you compile. Do not convert the main computation to float64.
6. Do not change, bypass or inspect the harness, tests or timing. Do not hardcode outputs.
7. Failing a check is a real result. Report it. Do not hide it.

| ID | Allowed techniques |
|---|---|
| C1 | C, compiler optimisation flags, CPU intrinsics. No external math library. |
| P1 | Python standard library only, scalar loops. |
| P4 | Numba with float32 arrays and compiled loops, nopython mode. No BLAS calls, including `np.dot` and `@`. NumPy only for arrays. |
| P3 | NumPy float32 arrays, native library calls allowed. |
| CB | C calling Apple Accelerate BLAS, the backend NumPy uses on this machine. Float32, same operation shapes. |
| CB2 | C with float32 arrays, Apple Accelerate library calls allowed. You may lay out the weights and group the library calls however you like. Matrix work must go through Accelerate: no hand-written matrix loops or CPU intrinsics for it. |
| MJ | Mojo 1.2.0 and its standard library, built with `mojo build`. SIMD and compile-time parameters allowed. No Python interop, no MAX, no external math library, no `parallelize`. Documentation and library source for this exact version are in `/Users/Shared/pvc-kit/mojo/docs`. |
