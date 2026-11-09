# Results

Pooled medians across official sittings (`results/raw/sitting-*.jsonl`, P2raw from `side-p2raw.jsonl`). Computed by `results/summarize.py`; full numbers in `results/summary.json`.

| ID | Version | decode tok/s (median) | min–max | runs | sittings | warm request s | first use s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| MJ | Mojo (bonus lane) | 139.655 | 136.436–140.049 | 20 | 1, 2 | 0.51 | 0.19 | 550 |
| C1 | Tuned C (SIMD allowed) | 138.384 | 128.346–138.794 | 30 | 1, 2, 3 | 0.52 | 0.15 | 452 |
| CB2 | C + Accelerate, free layout (rule added after sitting 1) | 136.683 | 134.075–136.933 | 20 | 2, 3 | 0.52 | 0.17 | 882 |
| P4 | Numba (no BLAS calls) | 133.671 | 132.586–134.145 | 20 | 1, 2 | 0.54 | 0.66 | 987 |
| P3 | NumPy (Accelerate allowed) | 121.034 | 117.799–121.757 | 30 | 1, 2, 3 | 0.59 | 0.28 | 530 |
| CB | C + Accelerate, "same operation shapes" rule | 77.287 | 69.524–77.575 | 30 | 1, 2, 3 | 0.93 | 0.2 | 451 |
| C0 | Karpathy's run.c, unchanged | 11.409 | 11.398–11.411 | 30 | 1, 2, 3 | 6.31 | 0.84 | 517 |
| P2 | PyPy, with a Claude-added PyPy path | 2.417 | 2.411–2.455 | 20 | 1, 2 | 29.77 | 3.85 | 620 |
| P1 | Plain Python (stdlib) | 0.439 | 0.438–0.439 | 20 | 1, 2 | 163.86 | 20.13 | 580 |
| P2raw | PyPy, same loop as P1 (side test, 3 runs) | 0.196 | 0.196–0.196 | 3 | side-p2raw | 367.46 | 45.72 | 602 |

## Paired comparisons

Ratio of the two versions inside the same block of the same sitting, then the median of those ratios. "ahead" = blocks where the first beat the second.

| pair | median ratio | ahead |
|---|---|---|
| MJ/C1 | 1.0092 | 20/20 |
| P4/C1 | 0.9668 | 4/20 |
| P3/C1 | 0.8753 | 0/30 |
| CB2/C1 | 0.9871 | 2/20 |
| CB2/P3 | 1.1272 | 20/20 |
| P3/CB | 1.5662 | 30/30 |
| P4/P3 | 1.1044 | 20/20 |
| C1/C0 | 12.1316 | 30/30 |
| P3/C0 | 10.6108 | 30/30 |
| C0/P1 | 25.9843 | 20/20 |
| C1/P1 | 315.1594 | 20/20 |
| P3/P1 | 275.6945 | 20/20 |
| P2/P1 | 5.5115 | 20/20 |

Model file: 438 MB of weights read per token; at C1's speed that is about 60.6 GB/s (derived, not a hardware-counter measurement).

## Claude usage (estimated at API list prices; the sessions ran on a Claude subscription)

| session | steps | estimated USD |
|---|---|---|
| P1 | initial, optimize_1, optimize_2, repair_1 | 4.19 |
| P3 | initial, optimize_1, optimize_2 | 5.01 |
| P4 | initial, optimize_1, optimize_2 | 6.69 |
| C1 | initial, optimize_1, optimize_2, optimize_2_redo, optimize_2_redo2 | 7.04 |
| CB | initial, optimize_1, optimize_2 | 2.80 |
| CB2 | initial, optimize_1, optimize_2, repair_1 | 7.48 |
| MJ | initial, optimize_1, optimize_2 | 7.86 |
| total | | 41.07 |
