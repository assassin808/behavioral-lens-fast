"""Experiment A analysis: AUC (positive = incorrect candidate) of each signal, bootstrap CI, per (model, traces) arm.

    python scripts/expA_analyze.py OUT_TAG ARM [ARM ...]      ARM = meas file stem without 'meas_' (e.g. base_on_gen-base)
Out: notes/expA_results_{OUT_TAG}.json, notes/fig/expA_pdoubt_{OUT_TAG}.png (refuses to overwrite).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import answer_like, grade_value
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/expA"
tag, arms = sys.argv[1], [a for a in sys.argv[2:] if "/" not in a]
pairs = [a.split("/") for a in sys.argv[2:] if "/" in a]   # "armA/armB": same traces, two models -> paired shift by correctness
OUT = os.environ.get("EXPA_OUT", f"{REPO}/notes")
oj, of = f"{OUT}/expA_results_{tag}.json", f"{OUT}/fig/expA_pdoubt_{tag}.png"
assert not os.path.exists(oj) and not os.path.exists(of), "refusing to overwrite"
SIG = ["p_doubt", "ratio", "p_doubt_plus_let", "p_doubt_ext", "ratio_ext", "u15_hs16", "u15_hs20", "u19_hs20", "real_recheck"]
rng = np.random.default_rng(0)


def auc(x, y):  # y = 1 incorrect; Mann-Whitney with ties
    x, y = np.asarray(x, float), np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    allv = np.concatenate([pos, neg]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = csum - (cnt - 1) / 2.0; r = avg[inv]
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(x, y, B=2000):
    x, y = np.asarray(x, float), np.asarray(y, int); n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); a = auc(x[i], y[i])
        if not np.isnan(a): v.append(a)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


_TXT = {}
def trace_text(traces, id_):
    """thinking text of a trace (for the repetition covariate)."""
    if traces not in _TXT:
        from expA_common import astra_rows, DSETS
        if traces.startswith("astra:"):
            rows = [r for ds in DSETS for r in astra_rows(traces.split(":")[1], ds)]
        else:
            rows = [json.loads(l) for l in open(f"{D}/gen_{traces}.jsonl")]
        _TXT[traces] = {r["id"]: r["text"].split("</think>")[0] for r in rows}
    return _TXT[traces][id_]


def rep_frac(prefix, n=30):
    """fraction of the last n non-empty lines before the candidate that already occurred verbatim earlier in the trace."""
    L = [l.strip() for l in prefix.split("\n") if l.strip()]
    if len(L) <= n: return 0.0
    seen = set(L[:-n]); last = L[-n:]
    return sum(l in seen for l in last) / n


def load(arm):   # arm may join several files with '+', e.g. base_on_base_p0+base_on_base_p1
    R = [json.loads(l) for a in arm.split("+") for l in open(f"{D}/meas_{a}.jsonl")]
    for r in R:
        r["p_doubt_plus_let"] = r["p_doubt"] + r["p_let"]; r["real_recheck"] = float(r["real_recheck"])
        r["ratio_ext"] = r["p_doubt_ext"] / max(r["p_doubt_ext"] + r["p_wrap_ext"], 1e-12)
        r["rep_frac"] = rep_frac(trace_text(r["traces"], r["id"])[: r["char_pos"]])
    R = [r for r in R if answer_like(r["value"])]          # amendment 3
    for r in R: r["correct_v1"] = r["correct"]; r["correct"] = grade_value(r["value"], r["gold"])   # amendment 4 (re-grade)
    first = {}
    for r in sorted(R, key=lambda r: r["cand_idx"]): first.setdefault((r["traces"], r["id"]), r["cand_idx"])
    for r in R: r["is_first"] = first[(r["traces"], r["id"])] == r["cand_idx"]
    return R


res = {}
for arm in arms:
    R = load(arm)
    for sub, sel in [("first", lambda r: r["is_first"]), ("first3", lambda r: True)]:
        S = [r for r in R if sel(r)]
        y = np.array([0 if r["correct"] else 1 for r in S])
        d = dict(n=len(S), n_correct=int((y == 0).sum()), n_incorrect=int(y.sum()))
        for s in SIG:
            if s not in S[0]: continue
            x = np.array([r[s] for r in S], float)
            a = auc(x, y); lo, hi = boot(x, y)
            d[s] = dict(auc=round(float(a), 3), ci=[round(lo, 3), round(hi, 3)],
                        median_correct=float(np.median(x[y == 0])) if (y == 0).any() else None,
                        median_incorrect=float(np.median(x[y == 1])) if y.any() else None,
                        mean_correct=float(x[y == 0].mean()) if (y == 0).any() else None,
                        mean_incorrect=float(x[y == 1].mean()) if y.any() else None)
        # confound checks: position in trace and dataset. AUC of log token position; within-dataset AUC; AUC of each signal after
        # regressing out log position + dataset (OLS residual).
        lp = np.log1p(np.array([r["tok_pos"] for r in S], float))
        DS = sorted({r["ds"] for r in S}); dsv = np.array([[r["ds"] == x for x in DS[1:]] for r in S], float).reshape(len(S), -1)
        d["log_tok_pos"] = dict(auc=round(float(auc(lp, y)), 3), ci=[round(v, 3) for v in boot(lp, y)])
        rp = np.array([r["rep_frac"] for r in S], float)
        d["rep_frac"] = dict(auc=round(float(auc(rp, y)), 3), mean_correct=float(rp[y == 0].mean()), mean_incorrect=float(rp[y == 1].mean()) if y.any() else None)
        X = np.column_stack([np.ones(len(S)), lp, dsv, rp])
        for s_ in ("p_doubt", "p_doubt_ext", "u15_hs16", "u15_hs20", "u19_hs20"):
            if s_ not in S[0]: continue
            x = np.array([r[s_] for r in S], float)
            resid = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
            d[s_]["auc_resid_pos_ds"] = round(float(auc(resid, y)), 3); d[s_]["ci_resid_pos_ds"] = [round(v, 3) for v in boot(resid, y)]
            for ds in DS:
                m = np.array([r["ds"] == ds for r in S])
                d[s_][f"auc_{ds}"] = round(float(auc(x[m], y[m])), 3); d[s_][f"n_inc_{ds}"] = int(y[m].sum()); d[s_][f"n_{ds}"] = int(m.sum())
        res[f"{arm}|{sub}"] = d
        print(arm, sub, d["n"], d["n_correct"], d["n_incorrect"], " ".join(f"{s}={d[s]['auc']}{d[s]['ci']}" for s in SIG if s in d))
        print("    pos", d["log_tok_pos"], "rep", d["rep_frac"]["auc"], "| resid(pos,ds,rep):", " ".join(f"{s}={d[s]['auc_resid_pos_ds']}{d[s]['ci_resid_pos_ds']} " + " ".join(f"{x}={d[s].get('auc_' + x)}" for x in ("gsm8k", "math500", "mathhard") if 'auc_' + x in d[s]) + f"" for s in ("p_doubt", "u15_hs16", "u15_hs20") if s in d))
for a, b in pairs:   # P3: p50c vs base on the same points
    A = {(r["id"], r["cand_idx"]): r for r in load(a)}; B = {(r["id"], r["cand_idx"]): r for r in load(b)}
    keys = [k for k in A if k in B and A[k]["is_first"]]
    d = {}
    for ok in (True, False):
        K = [k for k in keys if A[k]["correct"] == ok]
        for s_ in ("p_doubt", "p_doubt_ext", "ratio", "u15_hs16", "u15_hs20", "u19_hs20"):
            if s_ not in A[K[0]]: continue
            xa = np.array([A[k][s_] for k in K]); xb = np.array([B[k][s_] for k in K])
            d[f"{'correct' if ok else 'incorrect'}|{s_}"] = dict(n=len(K), median_a=float(np.median(xa)), median_b=float(np.median(xb)),
                median_ratio_b_over_a=float(np.median(xb) / max(np.median(xa), 1e-12)), median_paired_diff=float(np.median(xb - xa)),
                median_paired_logratio=float(np.median(np.log(np.clip(xb, 1e-6, None) / np.clip(xa, 1e-6, None)))) if s_.startswith("p_") or s_.startswith("ratio") else None)
    res[f"pair|{a}|{b}"] = d
    print("PAIR", a, "->", b, json.dumps(d, indent=0)[:3000])
json.dump(res, open(oj, "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, arm in zip(axes, arms):
    S = [r for r in load(arm) if r["is_first"]]
    for ok, col, lab in [(True, "#2a7ab9", "correct"), (False, "#d1495b", "incorrect")]:
        x = np.array([r["p_doubt"] for r in S if r["correct"] == ok])
        if len(x) == 0: continue
        jit = rng.uniform(-0.15, 0.15, len(x)) + (0 if ok else 1)
        ax.scatter(jit, np.clip(x, 1e-4, 1), s=8, alpha=0.5, color=col, label=f"{lab} (n={len(x)})")
        ax.hlines(np.median(x), (0 if ok else 1) - 0.25, (0 if ok else 1) + 0.25, color="k", lw=1.5)
    a = res[f"{arm}|first"]["p_doubt"]
    ax.set_title(f"{arm}\nAUC {a['auc']} [{a['ci'][0]}, {a['ci'][1]}]", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["correct", "incorrect"]); ax.set_yscale("log"); ax.legend(fontsize=6, frameon=False, loc="lower center")
axes[0].set_ylabel("P(doubt), next token")
fig.tight_layout(); os.makedirs(os.path.dirname(of), exist_ok=True); fig.savefig(of, dpi=150)
print("wrote", oj, of)
