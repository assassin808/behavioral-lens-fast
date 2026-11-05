"""Finding 1 from Astra's own first-sample logs (no generation): completion rate, accuracy conditional on completion, loop-end
rate, for unedited / p10c / p25c / p50c on each math set; plus paired accuracy on the problems both models completed."""
import glob, gzip, json, sys
sys.path.insert(0, "scripts")
from astra_eval import grade, loop_signals
RUNS = ["unedited", "p10c", "p25c", "p50c"]; SETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23", "aime24", "aime25"]
def load(run, ds):
    out = {}
    for f in sorted(glob.glob(f"astra_logs/exp8_math/eval/{run}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
        for l in gzip.open(f, "rt"):
            r = json.loads(l); t = r["text"]; done = "</think>" in t
            ls = loop_signals(t); ok, pred = grade(t, r["answer"]) if r.get("answer") is not None else (False, None)
            out[r["id"]] = dict(done=done, ok=ok, loop=ls["verbatim_loop_end"] or ls["repetitive_end"], verb=ls["verbatim_loop_end"], n=len(t))
    return out
res = {}
print("set | run | n | completion | acc_all | acc_on_completed | loop_end (verbatim) | acc_on_completed-in-both (n)")
for ds in SETS:
    D = {run: load(run, ds) for run in RUNS}
    if not D["unedited"]: continue
    both = [i for i in D["unedited"] if D["unedited"][i]["done"] and D["p50c"].get(i, {}).get("done")]
    for run in RUNS:
        d = D[run]; n = len(d)
        if not n: continue
        done = [v for v in d.values() if v["done"]]
        r = dict(n=n, completion=len(done)/n, acc_all=sum(v["ok"] for v in d.values())/n, acc_done=sum(v["ok"] for v in done)/max(1,len(done)),
                 loop=sum(v["loop"] for v in d.values())/n, verb=sum(v["verb"] for v in d.values())/n,
                 acc_both=sum(d[i]["ok"] for i in both if i in d)/max(1,len(both)), n_both=len(both))
        res[f"{ds}|{run}"] = r
        print(f"{ds} | {run} | {n} | {r['completion']:.2f} | {r['acc_all']:.3f} | {r['acc_done']:.3f} | {r['loop']:.2f} ({r['verb']:.2f}) | {r['acc_both']:.3f} ({r['n_both']})")
    # rescue analysis: p50c correct where unedited was wrong — were those unedited cut?
    resc = [i for i in D["p50c"] if D["p50c"][i]["ok"] and i in D["unedited"] and not D["unedited"][i]["ok"]]
    print(f"   p50c rescues {len(resc)}; of which unedited was cut: {sum(not D['unedited'][i]['done'] for i in resc)}; regressions: {sum(1 for i in D['unedited'] if D['unedited'][i]['ok'] and i in D['p50c'] and not D['p50c'][i]['ok'])}")
json.dump(res, open("notes/finding1_from_logs.json", "w"), indent=1)
