# What happened, in order (5 to 7 Nov 2026)

## Setup (5 Nov)
- Claude Opus 5.5, high effort, through Claude Code 2.1.318 on a Claude subscription, auto-updates off. Costs in `usage.json` are estimates at API list prices, not cash paid. Claude Code also makes small background calls on Haiku 5.5, so not every token is Opus.
- One model only: stories110M. 64 new tokens per timed request. stories15M was used only as a fast fixture for harness tests.
- Rules, prompts, harness, sandbox and recorder frozen (`bench/FROZEN.json`); `bench/freeze.py --check` before every session and sitting. Protocol versions are recorded in `bench/protocol.json`.
- Pilot runs (`results/raw/pilot-*.jsonl`, `smoke-final.jsonl`) were taken on battery and are never results.
- Mojo added as a bonus lane (protocol v2), with offline Mojo 1.2.0 docs because 1.2.0 is newer than the model's training data and sessions had no web.

## Recorder faults (5 Nov)
- C1: the recorder did not answer a "Do you want to overwrite" prompt, so `optimize_1` stopped at its last NOTES.md write (code work done) and `optimize_2` stopped after 65 s with no code change (void, kept on record). Fixed (protocol v3). C1's third step was rerun as `optimize_2_redo` (90 s, no code change: Claude treated it as finishing the interrupted pass), then once more as `optimize_2_redo2`, the last attempt, kept whatever the result. So C1 had extra steps; this is disclosed.
- P1: the recorder ended `initial` while a background check was running (code complete; the next step reran it and it passed). Fixed (protocol v4). The runner now archives and clears the sandbox user's /tmp between versions; logs show no cross-version reading.

## Hidden checks (5 Nov)
- Passed first time, exact stories: C1, CB, MJ, P3, P4.
- P1 failed on one token (token 3, a NUL byte: P1 printed `HEX 00` where C's string stops; INTERFACE.md never defined this). P2 failed because P1's loader crashed on PyPy. One repair step each (`prompts/repair.txt`); P1's final code is `repair_1`. Claude added a PyPy-only kernel during that work, so P2 is "PyPy with a Claude-added path"; P2raw is the side test without it.

## Sitting 1 (5 Nov 8:57 PM to 11:52 PM, plus the P2raw side test to 6 Nov 1:08 AM)
80 runs + 3 P2raw, all official, mains, no thermal warning, no failures, identical output hash for every run.
Rule flaw found afterwards: CB's rule said "same operation shapes"; NumPy (P3) had no such limit, re-laid its weights, and beat CB by 1.57x. The same-library comparison was not like for like. A new version, CB2, was added with the same freedom as P3 (rule added after results were seen; recorded in protocol.json).

## CB2 (6 Nov)
CB2's first attempts grouped several token positions into one call (batching), which the rules forbid; the rule wording allowed that reading. `optimize_2` void; one repair (`repair_1`) removed it; then it passed the hidden checks.

## Sitting 2 (6 Nov 1:39 PM to 4:30 PM)
90 runs, all 9 main versions including CB2, all official, no failures. Every version present in both sittings moved by less than 0.4%.

## Sitting 3 (7 Nov 10:48 AM to 11:18 AM)
50 runs: C0, C1, CB, CB2, P3 (the closest comparisons), 10 rounds each, all official, no failures, same output hash.

## Not measured
- Numba's cold compile time (P4 always loaded from a warm cache).
- Hardware memory-bandwidth counters. The ~62 GB/s single-core figure comes from Claude's own microbenchmarks in the session notes; the ~60 GB/s at C1's speed is derived from 438 MB x tokens/s.
- PyTorch (left out).
