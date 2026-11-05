import gzip, json, statistics, sys
for run in ["unedited", "p50c"]:
    f = f"astra_logs/exp8_math/eval/{run}/parts/gsm8k_s0of10/gsm8k_first_samples.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(f, "rt")]
    print(f"## {run}: {len(rows)} rows; keys={list(rows[0].keys())}")
    r = rows[0]
    for k, v in r.items():
        s = v if isinstance(v, str) else json.dumps(v)
        print(f"  {k}: {s[:200]!r}{' ...(%d chars)'%len(s) if len(s)>200 else ''}")
    tk = [k for k in r if isinstance(r[k], str) and len(r[k]) > 1000]
    for k in tk[:1]:
        L = [len(x[k]) for x in rows]
        closed = sum('</think>' in x[k] for x in rows)
        print(f"  text field '{k}': median chars {statistics.median(L):.0f}, max {max(L)}, closed </think> {closed}/{len(rows)}")
        x = max(rows, key=lambda x: len(x[k]))
        print("  TAIL of longest:", repr(x[k][-500:]))
m = json.load(open("astra_logs/exp8_math/eval/p50c/merged.json")); print(json.dumps(m, indent=1)[:1200])
