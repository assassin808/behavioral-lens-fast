"""Loop-onset analysis, stage 3 (CPU): class tables, nulls, predictions, AUCs.

    python scripts/onset_analyze.py OUT_JSON
Reads $BLF_DATA/onset/extract_{fresh,astra,hard}.jsonl and meas_{source}.jsonl (if present).
"""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np

D = f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/onset"
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
rng = np.random.default_rng(0)

X = {s: [json.loads(l) for l in open(f"{D}/extract_{s}.jsonl")] for s in ("fresh", "astra", "hard") if os.path.exists(f"{D}/extract_{s}.jsonl")}
NT = defaultdict(list)
for l in open(f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/nothink_runpod/base_nothink_runpod.jsonl"):
    r = json.loads(l); NT[r["id"]].append(r["correct"])
nothink_solved = {k: any(v) for k, v in NT.items()}

def onset(r, var):
    return {"primary": r["onset_tok"], "noquote": r["onset_noquote"], "terminal": r["onset_terminal"]}[var]

def stmts(r, det, var):
    if det == "answers" and var == "primary": return r["answers"]          # exactly as pre-registered (detector run on text[:onset])
    return r["answers_all"] if det == "answers" else r["results_all"]

def cls_at(lst, t):
    prev = [a for a in lst if a["tok_end"] <= t]
    if not prev: return "c", None
    return ("a" if prev[-1]["correct"] else "b"), t - prev[-1]["tok_end"]

def boot_ci(f, n, B=2000):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n); vals.append(f(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def table(rows, det, var):
    """one onset per looping trace (under variant var)."""
    on = [(r, onset(r, var)) for r in rows]
    on = [(r, t) for r, t in on if t is not None]
    res = dict(n_traces=len(rows), n_onsets=len(on))
    if not on: return res
    C, dist, nul_rows, uni = [], [], [], []
    for r, t in on:
        c, d = cls_at(stmts(r, det, var), t); C.append(c); dist.append(d)
        # matched-depth null: other traces of same source/ds, loop-free at depth t
        others = [o for o in rows if o["id"] != r["id"] and o["n_tok"] > t and (onset(o, var) is None or onset(o, var) > t)]
        cc = Counter(cls_at(stmts(o, det, var), t)[0] for o in others)
        tot = sum(cc.values()); nul_rows.append([cc[k] / tot if tot else np.nan for k in "abc"])
        # within-trace uniform null
        uu = Counter(cls_at(stmts(r, det, var), int(x))[0] for x in rng.integers(0, max(t, 1), 10))
        uni.append([uu[k] / 10 for k in "abc"])
    C = np.array(C); nul = np.array(nul_rows); uni = np.array(uni)
    cnt = {k: int((C == k).sum()) for k in "abc"}
    n = len(C); ab = cnt["a"] + cnt["b"]
    A = (C == "a").astype(float); B_ = (C == "b").astype(float); Cn = (C == "c").astype(float)
    ok = ~np.isnan(nul[:, 0])
    res.update(counts=cnt, share={k: cnt[k] / n for k in "abc"},
               P1_a_over_ab=cnt["a"] / ab if ab else None,
               P1_ci=boot_ci(lambda i: A[i].sum() / max(A[i].sum() + B_[i].sum(), 1), n) if ab else None,
               P2_c_share=cnt["c"] / n, P2_ci=boot_ci(lambda i: Cn[i].mean(), n),
               kill_bc_share=(cnt["b"] + cnt["c"]) / n,
               null_matched_share={k: float(np.nanmean(nul[:, j])) for j, k in enumerate("abc")},
               null_uniform_share={k: float(uni[:, j].mean()) for j, k in enumerate("abc")},
               P4_a_minus_null=float(A[ok].mean() - nul[ok, 0].mean()),
               P4_ci=boot_ci(lambda i: (A[ok][i] - nul[ok, 0][i]).mean(), int(ok.sum())),
               a_over_ab_null_matched=float(np.nanmean(nul[:, 0]) / max(np.nanmean(nul[:, 0]) + np.nanmean(nul[:, 1]), 1e-9)),
               dist_tok_median={k: (float(np.median([d for c, d in zip(C, dist) if c == k])) if (C == k).any() else None) for k in "ab"},
               dist_tok_q25_q75={k: ([float(np.percentile([d for c, d in zip(C, dist) if c == k], q)) for q in (25, 75)] if (C == k).any() else None) for k in "ab"},
               any_correct_before=float(np.mean([any(a["correct"] for a in stmts(r, det, var) if a["tok_end"] <= t) for r, t in on])),
               onset_tok_median=float(np.median([t for _, t in on])))
    lab_key = {"primary": "loop_label", "noquote": "label_noquote", "terminal": "label_terminal"}[var]
    res["loop_label_by_class"] = {k: dict(Counter(r.get(lab_key) for (r, t), c in zip(on, C) if c == k)) for k in "abc"}
    # difficulty: thinking-off solves
    for key, val in (("nothink_solved", True), ("nothink_unsolved", False)):
        sub = [c for (r, t), c in zip(on, C) if nothink_solved.get(r["id"]) is val]
        if sub: res[key] = dict(n=len(sub), **{k: sub.count(k) for k in "abc"})
    return res

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def auc_boot(pts, key="u15_hs16", B=2000):
    """positive = incorrect; bootstrap by trace."""
    by = defaultdict(list)
    for p in pts: by[p["id"]].append(p)
    ids = list(by)
    f = lambda P: auc([p[key] for p in P if not p["correct"]], [p[key] for p in P if p["correct"]])
    a = f(pts)
    if a is None: return dict(auc=None, n_corr=sum(p["correct"] for p in pts), n_inc=sum(not p["correct"] for p in pts))
    vals = []
    for _ in range(B):
        P = [p for i in rng.integers(0, len(ids), len(ids)) for p in by[ids[i]]]
        v = f(P)
        if v is not None: vals.append(v)
    return dict(auc=a, ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                n_corr=sum(p["correct"] for p in pts), n_inc=sum(not p["correct"] for p in pts), n_traces=len(ids))

def resid_logpos(pts, key="u15_hs16"):
    x = np.log1p([p["tok_pos"] for p in pts]); y = np.array([p[key] for p in pts])
    A = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return [dict(p, **{key + "_r": float(v)}) for p, v in zip(pts, y - A @ beta)]

R = dict(note="positive class for AUC = incorrect answer; higher u15_hs16 = 'doubt/incorrect' direction (Exp A)")
for s, rows in X.items():
    R[s] = {}
    for var in ("primary", "noquote", "terminal"):
        for det in ("answers", "results"):
            key = f"{var}|{det}"
            R[s][key] = {"all": table(rows, det, var)}
            for ds in sorted({r["ds"] for r in rows}):
                R[s][key][ds] = table([r for r in rows if r["ds"] == ds], det, var)
    R[s]["n_traces"] = len(rows); R[s]["n_loop"] = sum(r["loops"] for r in rows)
    R[s]["n_finish_noloop"] = sum(r["closed"] and not r["loops"] for r in rows)
    R[s]["onset_in_think_frac"] = float(np.mean([r["onset_in_think"] for r in rows if r["loops"]]))

# ---------------- hidden-state signal
for s in X:
    mf = f"{D}/meas_{s}.jsonl"
    if not os.path.exists(mf): continue
    M = resid_logpos([json.loads(l) for l in open(mf)])
    H = R[s]["auc"] = {}
    for g in ("loop_pre_onset", "finish", "cut_noloop"):
        P = [p for p in M if p["group"] == g]
        H[g] = auc_boot(P); H[g + "_logpos_resid"] = auc_boot(P, "u15_hs16_r")
        H[g + "_hs20_u15"] = auc_boot(P, "u15_hs20")
        H[g + "_median_u_corr_inc"] = [float(np.median([p["u15_hs16"] for p in P if p["correct"]])) if any(p["correct"] for p in P) else None,
                                       float(np.median([p["u15_hs16"] for p in P if not p["correct"]])) if any(not p["correct"] for p in P) else None]
    H["noloop_all"] = auc_boot([p for p in M if p["group"] != "loop_pre_onset"])
    H["loop_last_before_onset"] = auc_boot([p for p in M if p["group"] == "loop_pre_onset" and p["last_before_onset"]])
    # among correct answers only: does a loop-preceding correct answer look less 'correct' than a finishing one? (positive = loop)
    cl = [p["u15_hs16"] for p in M if p["group"] == "loop_pre_onset" and p["correct"]]
    cf = [p["u15_hs16"] for p in M if p["group"] == "finish" and p["correct"]]
    H["correct_only_loop_vs_finish"] = auc(cl, cf)
    cl = [p["u15_hs16_r"] for p in M if p["group"] == "loop_pre_onset" and p["correct"]]
    cf = [p["u15_hs16_r"] for p in M if p["group"] == "finish" and p["correct"]]
    H["correct_only_loop_vs_finish_logpos_resid"] = auc(cl, cf)
    H["by_ds"] = {ds: {g: auc_boot([p for p in M if p["group"] == g and p["ds"] == ds], B=500) for g in ("loop_pre_onset", "finish")}
                  for ds in sorted({p["ds"] for p in M})}

json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({s: {k: (v["all"] if isinstance(v, dict) and "all" in v else v) for k, v in R[s].items()} for s in X}, indent=1)[:20000])
