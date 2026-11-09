"""Writes bench/FROZEN.json: hashes of everything that defines the test. Run `python bench/freeze.py --check` before any AI session or official run."""
import glob, hashlib, json, sys
sys.path.insert(0, 'bench')
from workloads import eval_hash
F = ['RULES.md', 'CLAUDE.md', 'bench/INTERFACE.md', 'bench/protocol.json', 'bench/workloads.py', 'bench/client.py', 'bench/validate.py', 'bench/measure.py',
     'bench/check.py', 'bench/make_bundle.py', 'sandbox/session.sb.template', 'tools/record_session.py', 'tools/net_filter.py', 'ports/c0/adapter.c', 'sandbox/mojo-wrapper.sh'] + sorted(glob.glob('prompts/*.txt'))
now = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in F}
now['EVAL prompts'] = eval_hash()
if '--check' in sys.argv:
    old = json.load(open('bench/FROZEN.json'))['files']
    bad = [k for k in now if old.get(k) != now[k]]
    print('changed since freeze:', bad or 'nothing')
    sys.exit(1 if bad else 0)
d = {'version': 10, 'date': '2026-11-06', 'files': now}
d['protocol_hash'] = hashlib.sha256(json.dumps(now, sort_keys=True).encode()).hexdigest()
json.dump(d, open('bench/FROZEN.json', 'w'), indent=1)
print('frozen', d['protocol_hash'][:16], len(now), 'items')
