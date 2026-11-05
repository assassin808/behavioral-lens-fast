"""Test 1: recover u per layer, Astra's 64 columns, index-skew check, and a re-ranking of all 3584 neurons by loop selectivity
(from notes/q2_neuron_activations.json, d' of |act| loop vs productive) and by |uᵀw_j|.  Writes shared/."""
import json, numpy as np, torch
from scipy import stats
from safetensors.torch import load_file
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
base = load_file(f"{ROOT}/base/model.safetensors"); edit = load_file(f"{ROOT}/p50c/model.safetensors")
q2 = json.load(open("notes/q2_neuron_activations.json"))
out = {}
for L in [15, 19]:
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"; W0, W1 = base[n].float(), edit[n].float(); D = W1 - W0
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten(); U, S, Vt = torch.linalg.svd(D[:, cols], full_matrices=False); u = U[:, 0]
    # sign convention: +u = the direction the neurons wrote more of in loops (mean old projection of selected cols > 0)
    # sign: keep the SVD sign, which is the one used in Q2/Q3 (+u = loop direction: resid·u is higher inside loops)
    np.save(f"shared/u_L{L}.npy", u.numpy()); np.save(f"shared/astra_cols_L{L}.npy", cols.numpy())
    p = (u @ W0).numpy(); c = cols.numpy()
    d = np.array(q2["dprime"][str(L)])
    rank_d = np.argsort(-d); rank_p = np.argsort(-np.abs(p))
    pos_d = np.array([np.where(rank_d == j)[0][0] for j in c]); pos_p = np.array([np.where(rank_p == j)[0][0] for j in c])
    # index skew: are Astra's 64 columns biased toward low indices?  (Mann-Whitney vs uniform; and fraction below 1792)
    mw = stats.mannwhitneyu(c, np.setdiff1d(np.arange(3584), c), alternative="less")
    top64_d = rank_d[:64]; top64_p = rank_p[:64]
    # combined criterion: loop-selective AND writes u: score = d' * |uᵀw| (both standardized)
    z = lambda x: (x - x.mean()) / x.std(); comb = z(d) + z(np.abs(p)); rank_c = np.argsort(-comb); top64_c = rank_c[:64]
    np.save(f"shared/top64_by_dprime_L{L}.npy", top64_d); np.save(f"shared/top64_by_uproj_L{L}.npy", top64_p); np.save(f"shared/top64_combined_L{L}.npy", top64_c)
    out[L] = dict(astra_cols=c.tolist(), astra_cols_sorted=sorted(c.tolist()), median_index=float(np.median(c)), frac_below_half=float((c < 1792).mean()),
                  mannwhitney_p_low_index=float(mw.pvalue), overlap_top64_dprime=int(len(set(c) & set(top64_d))), overlap_top64_uproj=int(len(set(c) & set(top64_p))),
                  overlap_top64_combined=int(len(set(c) & set(top64_c))), astra_rank_by_dprime=pos_d.tolist(), astra_rank_by_uproj=pos_p.tolist(),
                  mean_uproj_astra=float(np.abs(p[c]).mean()), mean_uproj_top64_dprime=float(np.abs(p[top64_d]).mean()), mean_dprime_astra=float(d[c].mean()), mean_dprime_top64_uproj=float(d[top64_p].mean()))
    print(f"L{L}: Astra cols median index {np.median(c):.0f} (uniform: 1792), frac<1792 {(c<1792).mean():.2f}, MW p(low-skew) {mw.pvalue:.3f}; "
          f"sorted idx head {sorted(c.tolist())[:12]}")
    print(f"     overlap with top-64 by d' {len(set(c)&set(top64_d))}, by |uᵀw| {len(set(c)&set(top64_p))}, combined {len(set(c)&set(top64_c))}; "
          f"Astra rank by d' median {np.median(pos_d):.0f}, by |uᵀw| median {np.median(pos_p):.0f}")
    print(f"     |uᵀw|: Astra {np.abs(p[c]).mean():.4f}, top64-by-d' {np.abs(p[top64_d]).mean():.4f}; d': Astra {d[c].mean():.3f}, top64-by-|uᵀw| {d[top64_p].mean():.3f}")
json.dump(out, open("shared/test1_summary.json", "w"), indent=1); print("cos(u15,u19) =", float(np.dot(np.load("shared/u_L15.npy"), np.load("shared/u_L19.npy"))))
