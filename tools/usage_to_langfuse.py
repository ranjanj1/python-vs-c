"""Send one session's usage numbers to Langfuse, one span per step. Runs as Ranjan, outside the sandbox; no prompts, code or output are sent.
Costs are estimates at API list prices: subscription use is not billed this way.

  .venv/bin/python experiment/tools/usage_to_langfuse.py <session-id> [--dry-run]
"""
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sid = sys.argv[1].lower()
steps = []
for f in sorted((ROOT / 'sessions' / 'private' / sid).glob('*/usage.json')):
    ev = json.loads((f.parent / 'events.json').read_text())
    steps.append((f.parent.name, json.loads(f.read_text()), {k: ev.get(k) for k in ('end', 'seconds')}))
total = round(sum(u['estimated_usd_api_prices'] for _, u, _ in steps), 4)
if '--dry-run' in sys.argv:
    for step, u, ev in steps:
        print(step, ev, u['estimated_usd_api_prices'], u['models'])
    print(sid, len(steps), 'steps, not sent', total)
    sys.exit()

for line in (ROOT.parents[2] / '.env').read_text().splitlines():
    m = re.match(r'(LANGFUSE_(?:PUBLIC_KEY|SECRET_KEY|BASE_URL))=(.*)', line.strip())
    if m:
        os.environ[m[1]] = m[2].strip('"\'')
os.environ['LANGFUSE_HOST'] = os.environ['LANGFUSE_BASE_URL']

from langfuse import get_client, propagate_attributes  # noqa: E402

lf = get_client()
with propagate_attributes(session_id=sid):
    for step, u, ev in steps:
        with lf.start_as_current_observation(name=f'{sid}/{step}', as_type='span', metadata={**ev, 'api_calls': u['api_calls'], 'cost_basis': 'estimate at API list prices'}):
            with lf.start_as_current_observation(name='usage', as_type='generation', model=','.join(u['models'])) as g:
                g.update(usage_details={'input': u['input_tokens'], 'output': u['output_tokens'],
                                        'cache_read': u['cache_read_input_tokens'], 'cache_write': u['cache_creation_input_tokens']},
                         cost_details={'total': u['estimated_usd_api_prices']})
lf.flush()
print(sid, len(steps), 'steps sent', total)
