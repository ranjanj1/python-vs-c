"""Summary numbers computed from the raw official results.

  python3 results/summarize.py   ->  results/summary.json
Medians pool all runs of a version across sittings; ratios are per-block pairs inside one sitting, then the median of those pairs.
"""
import glob, json, statistics as st
from pathlib import Path

P = Path(__file__).resolve().parents[1]
rows = []
files = sorted(glob.glob(str(P / 'results/raw/sitting-*.jsonl')))
for f in files:
    rows += [json.loads(l) for l in open(f)]
assert all(r['official'] and r['exit'] == 0 and r['fixed_argmax_hash'] == rows[0]['fixed_argmax_hash'] for r in rows)
side = [json.loads(l) for l in open(P / 'results/raw/side-p2raw.jsonl')]
by = {}
for r in rows + side:
    by.setdefault(r['impl'], []).append(r)
versions = {}
for k, v in by.items():
    d = [x['decode_tok_s'] for x in v]
    versions[k] = {'runs': len(v), 'sittings': sorted({x['run'] for x in v}), 'tok_s': round(st.median(d), 3), 'min': round(min(d), 3), 'max': round(max(d), 3),
                   'request_s': round(st.median(x['request_s'] for x in v), 2), 'rss_mb': round(st.median(x['peak_rss_bytes'] for x in v) / 1e6),
                   'first_use_s': round(st.median(x['first_use_ttft_s'] for x in v), 2)}
blocks = {}
for r in rows:
    blocks.setdefault((r['run'], r['block']), {})[r['impl']] = r['decode_tok_s']


def pair(a, b):
    x = [v[a] / v[b] for v in blocks.values() if a in v and b in v]
    return {'median': round(st.median(x), 4), 'min': round(min(x), 4), 'max': round(max(x), 4), 'a_ahead': sum(i > 1 for i in x), 'rounds': len(x)}


pairs = {f'{a}/{b}': pair(a, b) for a, b in [('MJ', 'C1'), ('P4', 'C1'), ('P3', 'C1'), ('CB2', 'C1'), ('CB2', 'P3'), ('P3', 'CB'), ('P4', 'P3'), ('C1', 'C0'), ('P3', 'C0'), ('C0', 'P1'), ('C1', 'P1'), ('P3', 'P1'), ('P2', 'P1')]}
usd = {}
for f in glob.glob(str(P / 'sessions/*/*/usage.json')):
    k = f.split('/')[-3]
    if k in ('c1', 'p1', 'p3', 'p4', 'cb', 'cb2', 'mj'):
        usd[k] = usd.get(k, 0) + (json.load(open(f)).get('estimated_usd_api_prices') or 0)
out = {'sittings': [Path(f).stem for f in files], 'versions': versions, 'pairs': pairs, 'model_mb': 438,
       'gb_s_at_c1': round(versions['C1']['tok_s'] * .438, 1), 'estimated_usd': {k: round(v, 2) for k, v in usd.items()}, 'estimated_usd_total': round(sum(usd.values()), 2)}
(P / 'results/summary.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v['tok_s'] for k, v in versions.items()}), json.dumps({k: (v['median'], f"{v['a_ahead']}/{v['rounds']}") for k, v in pairs.items()}), out['gb_s_at_c1'], out['estimated_usd_total'])
