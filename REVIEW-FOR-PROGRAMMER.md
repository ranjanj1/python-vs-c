> Written and sent before any version was built or timed (early Nov 2026), to ask a programmer to look for anything unfair.
> Kept as it was sent. "AI" here means Claude Opus 5.5. The final method and results are in README.md, RESULTS.md and HISTORY.md.

# Please try to break this benchmark

I'm making a video comparing AI-written Python and C versions of llama2.c on an M1 MacBook. Comments on my last test said it was unfair, so please look for anything unfair or misleading before it runs. No AI versions or official results exist yet.

**Question:** how close can Python get to C on the same small language-model program, and what changes when both use the same math library?

**Versions**
- C0: Karpathy's run.c, pinned, unmodified, clang -O3, no fast-math
- C1: AI-tuned C loops, no math library
- P1: plain CPython, standard library only (Python floats are 64-bit, labelled as such)
- P4: Numba compiled loops, float32, no BLAS
- P3: NumPy float32 (Apple Accelerate here)
- CB: C calling the same Accelerate library
- P2, extra: the same P1 code on PyPy
- MJ, bonus: AI-written Mojo 1.2.0, tuned loops, no math library, no Python interop. It's reported separately because Mojo isn't Python, and C1 is its nearest comparison. This Mojo version is newer than the AI's training data, so its session gets the matching docs and library source offline.

**Method**
- Model: stories110M only, batch 1, greedy, 64 new tokens, one compute thread
- AI: Claude Opus 5.5, high effort, one session per version, three steps of up to 30 minutes, identical prompts
- Sandbox: no web and no other versions; hidden test prompts
- Correctness: token IDs, all 32,000 decoded tokens, logits within 5e-4 of the reference, including a 256-position sequence. Known-correct code was within 4.2e-5 of float64. A story that changes at a near-tie is reported as diverged, not failed.
- Timing: taken outside the program. First use, warm decode, fixed tokens and peak memory are reported separately; compile time has its own column
- Runs: at least 10 randomised blocks on each of 2 days; all raw samples published; confidence intervals on paired ratios

**Known limits:** one program, one AI, one machine, one attempt per version. It cannot say which language is fastest in general.

**What I'd like from you**
1. What would make you distrust the result?
2. Is any version favoured or handicapped?
3. Is the C baseline fair? Does single-thread make sense?
4. Is the Mojo lane fair, given the supplied docs?
5. Anything missing from the checks?

Full rules: RULES.md, EXPERIMENT-PLAN.md and bench/protocol.json.
