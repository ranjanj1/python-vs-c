# Interface contract (every implementation)

Start: `<program> <checkpoint.bin> <tokenizer.bin>`. After loading, print `READY`. Then read one command per line on stdin and answer on stdout.
Flush stdout after every line. All timing is done by the harness from outside; programs never report their own speed.

| Command | Answer |
|---|---|
| `ENCODE <text>` | `IDS <id> <id> ...` with BOS (1) first, no EOS |
| `PIECE <prev_id> <id>` | `HEX <hex bytes of the decoded piece>` |
| `GENERATE <max_new> <text>` | one `T <id>` line per new token as soon as it exists, then `DONE <count> <max\|stop\|context>` |
| `FIXED <id> <id> ...` | forward those ids at positions 0..n-1, full logits each step; `DONE <n> <argmax0> <argmax1> ...` |
| `LOGITS <path> <id> <id> ...` | as FIXED, then write all logits as little-endian float32, n x vocab, to path; `DONE <n>` |
| `QUIT` | exit 0 |

Every command starts from position 0 with a clean KV cache. Weights and compiled code stay loaded.
GENERATE is greedy (argmax, lowest index wins ties). Prompt tokens are forced, then new tokens are chosen.
It stops with `stop` if the chosen token is 1 (not printed), `max` after max_new tokens, `context` at seq_len positions.
`seq_len` and `vocab_size` come from the checkpoint header. A negative vocab_size in the header means unshared classifier weights.
