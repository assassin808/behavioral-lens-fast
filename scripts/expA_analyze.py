"""Experiment A analysis: AUC (positive = incorrect candidate) of each signal, bootstrap CI, per (model, traces) arm.

    python scripts/expA_analyze.py OUT_TAG ARM [ARM ...]      ARM = meas file stem without 'meas_' (e.g. base_on_gen-base)
Out: notes/expA_results_{OUT_TAG}.json, notes/fig/expA_pdoubt_{OUT_TAG}.png (refuses to overwrite).
"""
import json, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/expA"
tag, arms = sys.argv[1], sys.argv[2:]
oj, of = f"{REPO}/notes/expA_results_{tag}.json", f"{REPO}/notes/fig/expA_pdoubt_{tag}.png"
assert not os.path.exists(oj) and not os.path.exists(of), "refusing to overwrite"
SIG = ["p_doubt", "ratio", "p_doubt_plus_let", "u15_hs16", "u15_hs20", "u19_hs20", "real_recheck"]
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


def load(arm):
    R = [json.loads(l) for l in open(f"{D}/meas_{arm}.jsonl")]
    for r in R: r["p_doubt_plus_let"] = r["p_doubt"] + r["p_let"]; r["real_recheck"] = float(r["real_recheck"])
    return R


res = {}
for arm in arms:
    R = load(arm)
    for sub, sel in [("first", lambda r: r["cand_idx"] == 0), ("first3", lambda r: True)]:
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
        res[f"{arm}|{sub}"] = d
        print(arm, sub, d["n"], d["n_correct"], d["n_incorrect"], " ".join(f"{s}={d[s]['auc']}{d[s]['ci']}" for s in SIG if s in d))
json.dump(res, open(oj, "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, arm in zip(axes, arms):
    S = [r for r in load(arm) if r["cand_idx"] == 0]
    for ok, col, lab in [(True, "#2a7ab9", "correct"), (False, "#d1495b", "incorrect")]:
        x = np.array([r["p_doubt"] for r in S if r["correct"] == ok])
        if len(x) == 0: continue
        jit = rng.uniform(-0.15, 0.15, len(x)) + (0 if ok else 1)
        ax.scatter(jit, np.clip(x, 1e-4, 1), s=8, alpha=0.5, color=col, label=f"{lab} (n={len(x)})")
        ax.hlines(np.median(x), (0 if ok else 1) - 0.25, (0 if ok else 1) + 0.25, color="k", lw=1.5)
    a = res[f"{arm}|first"]["p_doubt"]
    ax.set_title(f"{arm}\nAUC {a['auc']} [{a['ci'][0]}, {a['ci'][1]}]", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["correct", "incorrect"]); ax.set_yscale("log"); ax.legend(fontsize=6, frameon=False, loc="lower center")
axes[0].set_ylabel("P(doubt) of next token after first candidate answer")
fig.tight_layout(); os.makedirs(os.path.dirname(of), exist_ok=True); fig.savefig(of, dpi=150)
print("wrote", oj, of)
