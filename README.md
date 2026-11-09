# How close can Python get to C? The experiment

Code, prompts, rules and raw results for an It's Not Magic film. Claude Code (Claude Opus 5.5, high effort) rebuilt the inference
program of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (`run.c`, pinned commit in `sources/ORIGIN.md`) in several
languages and styles. A separate harness, which Claude never touched, timed every version on one Apple M1 Mac with 8 GB of memory,
one thread, model stories110M (438 MB of float32 weights), 64 new tokens per request.

**Status: finished.** Three official sittings (5, 6 and 7 Nov 2026), 220 official runs plus a 3-run PyPy side test.
Start with `RESULTS.md` (numbers), then `HISTORY.md` (what happened, in order, including every failure, repair and rule change).

## The versions
| ID | What it is | Who wrote it |
|---|---|---|
| C0 | Karpathy's `run.c`, unchanged, built with his Makefile default (`-O3`) | Karpathy (`ports/c0/adapter.c` wraps it for the harness) |
| P1 | Plain Python, standard library only | Claude |
| P2 | P1's code run on PyPy; Claude also added a PyPy-only path | Claude |
| P2raw | Side test: P1's loop on PyPy with that path switched off (one labelled human edit, `ports/p2raw/run.py:278`) | Claude + one human line |
| P3 | NumPy (native library calls allowed; NumPy uses Apple Accelerate here) | Claude |
| P4 | Numba (compiled loops, no BLAS calls) | Claude |
| C1 | Tuned C (SIMD allowed) | Claude |
| CB | C calling Apple Accelerate, with "same operation shapes" | Claude |
| CB2 | C calling Apple Accelerate, free weight layout (added after a rule flaw was found) | Claude |
| MJ | Mojo 1.2.0, no Python and no outside maths library (bonus lane) | Claude |

## Results in one table (median decode tokens/s; full tables, paired ratios and costs in `RESULTS.md`)
| MJ | C1 | CB2 | P4 | P3 | CB | C0 | P2 | P1 | P2raw |
|---|---|---|---|---|---|---|---|---|---|
| 139.655 | 138.384 | 136.683 | 133.671 | 121.034 | 77.287 | 11.409 | 2.417 | 0.439 | 0.196 |

## What changed since the last update (sitting 3, 7 Nov 2026)
- `results/raw/sitting-3.jsonl`: 50 official runs of C0, C1, CB, CB2 and P3 (the closest comparisons), 10 rounds each, mains power,
  no thermal warning, no failures, same output hash as every other run.
- `results/summarize.py` and `results/summary.json`: recompute every number in `RESULTS.md` from the raw files, pooled over all sittings.
- `RESULTS.md`, `HISTORY.md`: the final numbers and the full account.
- `snapshots/`: the code after every Claude step (source and notes; compiled binaries left out).
- `sessions/`: every Claude session behind every version: terminal recording (`session.cast`, asciicast-style events with no header line; offsets are seconds from the start), plain transcript, Claude Code's own session log (`claude-session.jsonl`; Claude Code's internal API-request captures, system-prompt snapshots and tool/skill listings removed, the conversation kept), usage estimate, recorder events and the sandbox's network log (only `api.anthropic.com` allowed). Plus the exact prompt sent for each step and the runner logs (`sessions/logs/`).
- The compiled binaries that were timed (`ports/*`), except the Mojo binary, which embedded a local build path (rebuild it with `mojo build ports/mj/mj.mojo -o ports/mj/mj`), and the synthetic test checkpoints in `ports/p3/prof/synth/`.

## How each version was made
One fresh Claude Code session per version, sandboxed (`sandbox/`), no web, no access to other versions. Three prompts (`prompts/`):
`initial.txt` (build it, pass the tests), `optimize.txt` (profile and speed up), then `optimize.txt` again, word for word; 30 minutes each.
`repair.txt` was used only where a hidden check failed (see `HISTORY.md`). Rules every version had to follow: `RULES.md`, `CLAUDE.md`, `bench/INTERFACE.md`.
Each step's final code is in `snapshots/<id>/<step>/`; the code that was timed is in `ports/<id>/`.

## How it was checked and timed
- `bench/validate.py`: hidden prompts Claude never saw. Eligible = logits within tolerance (5e-4) of the reference; the report also says whether
  the generated stories matched exactly. Reports in `validation/`. Every timed version passed and matched.
- `bench/measure.py`, `bench/client.py`: timing from outside the program. Per run: startup, first-use time to first token, warm request
  (after startup and warm-up), decode tokens/s (63 gaps between 64 tokens / seconds from first to last token), peak RSS, CPU/wall.
- `bench/sitting.sh`: one official sitting = every version once per block in a shuffled order (fixed seed per sitting), 30 s rest between runs.
  Refuses to start off mains power, stops on a thermal warning, and `bench/freeze.py --check` must pass (hashes in `bench/FROZEN.json`).
- Compile time is outside every timer: C is built before timing, Numba loads from its warm cache. One-off build times: `validation/build-times.txt`.

## How to review it (a short checklist)
1. Recompute the numbers: `python3 results/summarize.py`, then compare `results/summary.json` with `RESULTS.md`. Every row in
   `results/raw/sitting-*.jsonl` is official, exited 0 and has the same `fixed_argmax_hash` (the script asserts this).
2. Read the rules and prompts (`RULES.md`, `prompts/`), then a version's notes (`ports/<id>/NOTES.md`), its per-step code (`snapshots/<id>/`)
   and what Claude actually did in each step (`sessions/<id>/<step>/transcript.txt` and `claude-session.jsonl`).
3. Check correctness: `validation/<ID>-eval.json` (and `validation/superseded/` for the two reports that failed before their repair).
4. Read `HISTORY.md` for the disclosures: recorder faults and C1's extra steps, the P1/P2 repairs and the Claude-added PyPy path,
   the CB rule flaw and the later CB2 version, CB2's batching repair, and what was not measured.

## How to replicate it
You need an Apple Silicon Mac (results are for one M1, 8 GB; another machine will give different numbers).
1. Toolchain (versions in `bench/protocol.json`): Apple clang 21, CPython 3.12.13 with NumPy 2.5.4 (Accelerate) and Numba 0.68.1,
   PyPy 7.3.21, Mojo 1.2.0. The harness expects them as separate offline kits under `/Users/Shared/pvc-kit` (paths in `bench/sitting.sh`);
   put yours there or edit those paths.
2. Inputs: `stories110M.bin` from https://huggingface.co/karpathy/tinyllamas into `models/`, and `tokenizer.bin` from llama2.c at the pinned
   commit into `sources/upstream/`. SHA-256 for both is in `bench/protocol.json`.
3. Build the compiled versions (commands in `validation/build-times.txt` and each `build.sh`): `ports/c1/build.sh`, `ports/cb2/build.sh`,
   `clang -O2 -Wall -Wextra -o ports/cb/cb ports/cb/cb.c -framework Accelerate`, `mojo build ports/mj/mj.mojo -o ports/mj/mj`.
   C0 is Karpathy's `run.c` included through `ports/c0/adapter.c`, built with `-O3` and no fast-math (`bench/protocol.json`, "reference").
4. Check correctness: `python bench/validate.py --name <ID> --set eval -- <command>` from the repo root (usage at the top of the file).
5. Time it: `bench/sitting.sh 1`, `bench/sitting.sh 2` and `bench/sitting.sh 3` on different days, on mains power with other apps closed.
   Each refuses to overwrite an existing sitting file, so move ours aside first. Then `python3 results/summarize.py`.
6. To rebuild the versions themselves with Claude, use the same prompts, rules and sandbox (`tools/`, `sandbox/`); a new run will
   produce different code and different numbers, which is part of what such a test measures.

## Folders
| Folder | Contents |
|---|---|
| `results/raw/` | Every official run (`sitting-1/2/3.jsonl`), the PyPy side test, and pilots (pilots are never results) |
| `results/summarize.py`, `results/summary.json` | Recompute every number in `RESULTS.md` |
| `ports/`, `snapshots/` | Timed code and per-step code (with the compiled binaries) |
| `sessions/` | Every Claude session: recordings, transcripts, session logs, usage, network logs, prompts |
| `bench/`, `controls/`, `sandbox/`, `tools/` | Harness, control programs, sandbox, recorder and step runner |
| `validation/` | Hidden-prompt check reports |
| `sources/upstream/` | llama2.c, MIT licence, unmodified (tokenizer not included) |

`snapshots/t9`, `snapshots/t10` and the `bg-check`, `mojo-check`, `overwrite-check`, `t9`, `t10` usage files are recorder tests, not results.

Not included: the model weights (`models/stories110M.bin`, 438 MB; download link above), `tokenizer.bin` (its reuse terms still need their own
check, `sources/ORIGIN.md`; it ships with llama2.c at the pinned commit), the Python/PyPy/Mojo environments (`bench/kit`, `/Users/Shared/pvc-kit`),
and video copies made for the film (`session.mp4`); the original `.cast` recordings are included.

Note: `bench/protocol.json` was frozen with `"sittings": 2` in its measurement section; sitting 3 (C versions and NumPy only) was added
afterwards for the closest comparisons and is described in `HISTORY.md`.

## Licence
MIT for this repo's own code (LICENSE). Karpathy's llama2.c, included under `sources/upstream/` and every `reference/`, keeps its MIT licence.
