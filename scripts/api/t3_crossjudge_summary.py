"""Summarise the cross-family re-judging (notes/api_judge_crossfamily.md).
    python scripts/api/t3_crossjudge_summary.py <out.json>"""
import json, sys, os, random
import numpy as np
from scipy.stats import spearmanr
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1]; assert not os.path.exists(OUT)
RUNS = {("sonnet", "sonnet", "orig"): "notes/test3_autointerp_32768_claude-sonnet-5-5_killarney.json",
        ("sonnet", "sonnet", "retest"): "notes/api_t3_sonnet__claude-sonnet-5-5_desc-desc_shuf.json",
        ("sonnet", "gpt-oss"): "notes/api_t3_sonnet__gpt-oss-120b.json",
        ("gpt-oss", "gpt-oss"): "notes/api_t3_gpt-oss-120b__gpt-oss-120b.json",
        ("gpt-oss", "sonnet"): "notes/api_t3_gpt-oss-120b__claude-sonnet-5-5_desc-desc_shuf.json"}
G = {"astra_L15": lambda x: not x["is_random"] and x["layer"] == 15, "astra_L19": lambda x: not x["is_random"] and x["layer"] == 19,
     "random": lambda x: x["is_random"]}
D = {k: {(x["layer"], x["unit"]): x for x in json.load(open(f"{REPO}/{v}"))["neurons"]} for k, v in RUNS.items() if os.path.exists(f"{REPO}/{v}")}
KEYS = ["ap_desc_negm", "ap_desc_negn", "ap_desc_shuf_negm", "ap_desc_shuf_negn", "ap_desc_seg_negm", "ap_desc_loop_negn", "ap_desc_loop_negm"]
def mean_ci(v):
    v = [a for a in v if a is not None]; rng = random.Random(0)
    b = [np.mean([v[rng.randrange(len(v))] for _ in v]) for _ in range(2000)]
    return [float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
res = {"tables": {}, "diffs": {}, "corr": {}}
for k, d in D.items():
    res["tables"]["|".join(k)] = {g: {m: mean_ci([x.get(m) for x in d.values() if f(x)]) for m in KEYS if any(x.get(m) is not None for x in d.values())} for g, f in G.items()}
def paired(k1, k2, m, g):
    a, b = D[k1], D[k2]; ids = [i for i in a if i in b and G[g](a[i]) and a[i].get(m) is not None and b[i].get(m) is not None]
    return mean_ci([a[i][m] - b[i][m] for i in ids]) + [len(ids)]
for (k1, k2, lab) in [(("sonnet", "gpt-oss"), ("sonnet", "sonnet", "orig"), "P1: Sonnet desc, gpt-oss judge - Sonnet judge"),
                      (("sonnet", "sonnet", "retest"), ("sonnet", "sonnet", "orig"), "Sonnet judge retest - orig"),
                      (("gpt-oss", "sonnet"), ("sonnet", "sonnet", "orig"), "Sonnet judge: gpt-oss desc - Sonnet desc"),
                      (("gpt-oss", "gpt-oss"), ("sonnet", "gpt-oss"), "gpt-oss judge: gpt-oss desc - Sonnet desc")]:
    if k1 in D and k2 in D:
        res["diffs"][lab] = {g: {m: paired(k1, k2, m, g) for m in ["ap_desc_negm", "ap_desc_negn", "ap_desc_shuf_negn"]} for g in G}
for k in D:   # P2 desc - shuffled, P3 astra - random
    d = D[k]; r = {}
    for g in G:
        r[f"desc_minus_shuf_negn_{g}"] = mean_ci([x["ap_desc_negn"] - x["ap_desc_shuf_negn"] for x in d.values() if G[g](x) and x.get("ap_desc_shuf_negn") is not None])
    for g in ["astra_L15", "astra_L19"]:
        for m in ["ap_desc_negm", "ap_desc_negn"]:
            a = [x[m] for x in d.values() if G[g](x)]; b = [x[m] for x in d.values() if G["random"](x)]; rng = random.Random(0)
            bs = [np.mean([a[rng.randrange(len(a))] for _ in a]) - np.mean([b[rng.randrange(len(b))] for _ in b]) for _ in range(2000)]
            r[f"{g}_minus_random_{m}"] = [float(np.mean(a) - np.mean(b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["diffs"]["P2P3|" + "|".join(k)] = r
if ("sonnet", "gpt-oss") in D:   # per-neuron agreement of the two judges on the same descriptions
    a, b = D[("sonnet", "sonnet", "orig")], D[("sonnet", "gpt-oss")]
    for m in ["ap_desc_negm", "ap_desc_negn"]:
        ids = [i for i in a if i in b]; res["corr"][m] = float(spearmanr([a[i][m] for i in ids], [b[i][m] for i in ids])[0])
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
