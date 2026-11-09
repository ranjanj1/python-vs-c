#!/bin/sh
# usage: profrun.sh <src.c> ; runs the fixture GENERATE workload under the -DPROF build
cc -O3 -mcpu=native -DPROF -o /tmp/cb2t/cb2prof "$1" -framework Accelerate || exit 1
python3 -c "
import json;fx=json.load(open('/Users/agentlab/pvc/cb2/fixtures.json'))
print(''.join(f\"GENERATE 64 {p['text']}\n\" for p in fx['prompts'])+'QUIT')" | /tmp/cb2t/cb2prof /Users/agentlab/pvc/cb2/model.bin /Users/agentlab/pvc/cb2/tokenizer.bin >/dev/null
