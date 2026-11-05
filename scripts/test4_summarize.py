"""Test 4: markdown tables + figure from test4_analyze.py output and the label files.
    python scripts/test4_summarize.py ANALYZE.json LABELS_SONNET.json [LABELS_OPUS.json] FIGPREFIX
"""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = json.load(open(sys.argv[1])); LS = json.load(open(sys.argv[2]))["labels"]
LO = json.load(open(sys.argv[3]))["labels"] if len(sys.argv) > 4 else None
FP = sys.argv[-1]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.6})


def lab(nm, j):
    v = LS.get(f"{nm}:{j}"); return (v["label"], v["repetition_related"], v["score"] and v["score"]["bal_acc"]) if v else ("(unlabelled)", None, None)


def ex(nm, j, n=2):
    e = A[nm]["exemplars"].get(str(j), {}).get("top", [])[:n]
    return " ⏐ ".join((x["pre"][-60:] + "[[" + x["tok"] + "]]" + x["post"][:12]).replace("\n", "\\n").replace("|", "¦") for x in e)


print("## paired:", json.dumps(A["paired"]), "\nrecon FVU at loop positions (base / edit acts):", json.dumps(A["recon_fvu_loop"]))
for nm in ["ours16", "pub18", "ours20"]:
    r = A[nm]
    print(f"\n### {nm} (hidden_states[{r['layer_hs']}], d_sae {r['d_sae']}, read against {r['u']}) cfg {json.dumps(r['cfg'])}")
    print(f"max |cos(dec, u)| = {r['max_abs_cos_u']:.3f}; #features |cos|>0.3: {r['n_feats_cos_gt_0p3']}; "
          f"residual ‖u − proj on top-k greedy features‖ k=1..10: {[round(x, 2) for x in r['omp_resid_norm']]}; "
          f"cos(mean Δresid at loop pos, u) = {r['cos_u_dresid']:+.2f}; frac of top-20 edit-ON features in base top-100 productive: {r['frac_on20_in_prodtop100']:.2f}")
    for g in ["cos_plus_u", "cos_minus_u", "contrib_u_loop_minus_prod", "edit_off_loop", "edit_on_loop", "cos_dresid", "loop_selective", "prod_selective"]:
        print(f"\n**{g}**\n\n| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |\n|---|---|---|---|---|---|---|---|---|")
        for j in r["groups"][g][:10]:
            f = r["feats"][str(j)]; l, rep, sc = lab(nm, j)
            print(f"| {j} | {l} | {'Y' if rep else ('N' if rep is not None else '')} | {sc if sc is None else round(sc, 2)} | {f['cos_u']:+.2f} | {f['dprime']:+.2f} | "
                  f"{f['edit_base_loop']:.2f}→{f['edit_edit_loop']:.2f} | {f['edit_freq_base_loop']:.2f}→{f['edit_freq_edit_loop']:.2f} | {ex(nm, j)} |")
# predictions
print("\n### Prediction tallies")
for nm in ["ours16", "pub18", "ours20"]:
    r = A[nm]; top5 = r["groups"]["cos_plus_u"][:5]; off10 = r["groups"]["edit_off_loop"][:10]
    rep5 = sum(bool(lab(nm, j)[1]) for j in top5); rep10 = sum(bool(lab(nm, j)[1]) for j in off10)
    print(f"- {nm}: P4a max|cos|={r['max_abs_cos_u']:.2f} (<0.5?) ; rep-labelled in top-5 +cos: {rep5}/5 ; P4b rep-labelled in top-10 OFF: {rep10}/10 ; "
          f"P4c top-20 ON in prod-top100: {r['frac_on20_in_prodtop100']:.2f}")
allsc = [v["score"]["bal_acc"] for v in LS.values() if v.get("score")]
print(f"\nSonnet detection balanced accuracy: mean {np.mean(allsc):.2f}, median {np.median(allsc):.2f}, n={len(allsc)}")
if LO:
    print("\n### Opus spot-check vs Sonnet\n\n| feature | sonnet label | opus label | rep? S/O | bal.acc S/O |\n|---|---|---|---|---|")
    agree = []; dS = []; dO = []
    for k, v in LO.items():
        s = LS.get(k)
        if not s: continue
        agree.append(s["repetition_related"] == v["repetition_related"])
        if s.get("score") and v.get("score"): dS.append(s["score"]["bal_acc"]); dO.append(v["score"]["bal_acc"])
        print(f"| {k} | {s['label']} | {v['label']} | {'Y' if s['repetition_related'] else 'N'}/{'Y' if v['repetition_related'] else 'N'} | "
              f"{s['score'] and round(s['score']['bal_acc'], 2)}/{v['score'] and round(v['score']['bal_acc'], 2)} |")
    print(f"\nrepetition-related flag agreement {np.mean(agree):.2f} (n={len(agree)}); mean det. bal.acc Sonnet {np.mean(dS):.2f} vs Opus {np.mean(dO):.2f}; "
          f"Pearson {np.corrcoef(dS, dO)[0, 1]:.2f}")

# figure: per SAE, scatter of edit change at loop positions vs cos(dec,u)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for k, nm in enumerate(["ours16", "pub18", "ours20"]):
    r = A[nm]; F = r["feats"]
    xs = np.array([F[j]["cos_u"] for j in F]); ys = np.array([F[j]["edit_diff_loop"] for j in F])
    rep = np.array([bool(lab(nm, int(j))[1]) for j in F])
    ax[k].scatter(xs[~rep], ys[~rep], s=14, color=C[3], alpha=0.7, label="other label", edgecolor="white", linewidth=0.5)
    ax[k].scatter(xs[rep], ys[rep], s=18, color=C[1], label="repetition-related label", edgecolor="white", linewidth=0.5)
    ax[k].axhline(0, color="#888", lw=0.8); ax[k].axvline(0, color="#888", lw=0.8)
    ax[k].set_title(f"{nm} (hs[{r['layer_hs']}]), selected features"); ax[k].set_xlabel(f"cos(decoder, {r['u']})")
    if k == 0: ax[k].set_ylabel("Δ mean activation at loop pos. (edit − base)"); ax[k].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FP + "_features.png", dpi=150); print("\nfigure:", FP + "_features.png")
