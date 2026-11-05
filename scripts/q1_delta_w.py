"""Q1: what was changed?  ΔW per-column norms, SVD of ΔW, logit lens of the write vectors.

    python scripts/q1_delta_w.py <base.safetensors> <edited.safetensors> <out.json>
"""
import json, sys
import numpy as np
import torch
from safetensors.torch import load_file

base_p, edit_p, out_p = sys.argv[1:4]
base = load_file(base_p)
edit = load_file(edit_p)
E = base["model.language_model.embed_tokens.weight"].float()          # tied unembedding [V, d]
nw = base["model.language_model.norm.weight"].float()                  # final RMSNorm scale
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data2/ec2-user/models/qwen3.5-0.8b-astra/base")

def lens(v, k=15):
    v = v / (v.norm() + 1e-6)
    logits = E @ (nw * v)
    top = torch.topk(logits, k).indices.tolist(); bot = torch.topk(-logits, k).indices.tolist()
    return [tok.decode([i]) for i in top], [tok.decode([i]) for i in bot]

res = {}
for name in base:
    if not torch.equal(base[name], edit[name]):
        W0, W1 = base[name].float(), edit[name].float()
        D = W1 - W0
        cols = torch.nonzero(D.abs().sum(0) > 0).flatten()
        Dc, W0c, W1c = D[:, cols], W0[:, cols], W1[:, cols]
        cos = torch.nn.functional.cosine_similarity(W0c, W1c, dim=0)
        S = torch.linalg.svdvals(Dc)
        # low-rank summary: how much energy in top-k singular directions
        energy = (S**2).cumsum(0) / (S**2).sum()
        U, _, Vt = torch.linalg.svd(Dc, full_matrices=False)
        # mean Δ direction across the 64 columns
        mean_delta = Dc.mean(1)
        # does the edit push toward or away from the mean old write direction?
        r = {
            "n_cols": len(cols), "cols": cols.tolist(),
            "rel_change_fro": (D.norm() / W0.norm()).item(),
            "col_delta_norm": Dc.norm(dim=0).tolist(), "col_old_norm": W0c.norm(dim=0).tolist(), "col_new_norm": W1c.norm(dim=0).tolist(),
            "cos_old_new": cos.tolist(),
            "svdvals": S.tolist(), "energy_cum": energy.tolist(),
            "rank_90pct": int((energy < 0.9).sum().item() + 1),
            "mean_delta_cos_with_top_sv": torch.nn.functional.cosine_similarity(mean_delta, U[:, 0], dim=0).item(),
            "lens_top_sv1": lens(U[:, 0]), "lens_top_sv2": lens(U[:, 1]), "lens_mean_delta": lens(mean_delta),
            "lens_mean_old_col": lens(W0c.mean(1)), "lens_mean_new_col": lens(W1c.mean(1)),
        }
        res[name] = r
        print(f"\n### {name}")
        print(f"cols={len(cols)}  rel_fro={r['rel_change_fro']:.4f}  rank90%={r['rank_90pct']}  top5 svdvals={[round(s,3) for s in S[:5].tolist()]}  energy top1/2/4={[round(e,3) for e in energy[[0,1,3]].tolist()]}")
        print(f"cos(old,new) per col: mean={cos.mean():.3f} min={cos.min():.3f} max={cos.max():.3f}  (#cols with cos<0: {(cos<0).sum().item()})")
        print(f"||Δ||/||old|| per col: mean={(Dc.norm(dim=0)/W0c.norm(dim=0)).mean():.3f}  ||new||/||old|| mean={(W1c.norm(dim=0)/W0c.norm(dim=0)).mean():.3f}")
        print(f"cos(meanΔ, U1)={r['mean_delta_cos_with_top_sv']:.3f}")
        for k in ["lens_top_sv1", "lens_mean_delta", "lens_mean_old_col", "lens_mean_new_col"]:
            print(f"  {k}: +{r[k][0]}\n  {' '*len(k)}  -{r[k][1]}")
json.dump(res, open(out_p, "w"), indent=1)
