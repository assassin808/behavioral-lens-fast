"""Exp B exploratory (NOT pre-registered): position-matched PR rate (segments starting in a word window), and excerpts."""
import gzip, json, sys, random, numpy as np
R = [json.loads(l) for l in gzip.open("/scratch/yang0531/blf_data/expB/texts_full.jsonl.gz", "rt")]
def rate(r, lo, hi, sim, tau):
    k = 2 if sim == "cos" else 3; by = {}
    for p in r["pairs"]: by.setdefault(p[0], []).append(p)
    el = hit = 0
    for i in range(11, r["n_seg"]):
        if not (lo <= r["seg_start_word"][i] < hi): continue
        el += 1; hit += any(i - 11 <= p[1] <= i - 2 and p[k] >= tau and not p[4] for p in by.get(i, []))
    return hit / el if el else None
G = lambda r: r["group"] + " | " + (r["cls"] if r["group"] in ("p50c_think", "base_think_preloop") else "ALL")
out = {}
for lo, hi in [(0, 2000), (2000, 5000), (5000, 10000)]:
    for sim, tau in [("cos", 0.7), ("lex", 0.5)]:
        rows = {}
        for r in R:
            v = rate(r, lo, hi, sim, tau)
            if v is not None: rows.setdefault(G(r), []).append(v)
        out[f"{sim}>={tau} words[{lo},{hi})"] = {k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(rows.items())}
json.dump(out, open("/scratch/yang0531/blf_data/expB/explore_position.json", "w"), indent=1)
for k, v in out.items():
    print("\n" + k)
    for g, (n, m) in v.items(): print(f"  {g:45s} n={n:4d} PR={m}")
