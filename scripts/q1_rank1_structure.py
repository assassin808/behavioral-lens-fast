"""Q1b: ΔW is rank-1 per layer: ΔW = u aᵀ (a sparse over 64 columns). Is a_j ∝ uᵀ w0_j (projection removal / reflection)?
And is u the same direction in L15 and L19?"""
import torch, json, sys
from safetensors.torch import load_file
base = load_file("/data2/ec2-user/models/qwen3.5-0.8b-astra/base/model.safetensors")
edit = load_file("/data2/ec2-user/models/qwen3.5-0.8b-astra/p50c/model.safetensors")
us = {}
out = {}
for L in [15, 19]:
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    W0, W1 = base[n].float(), edit[n].float()
    D = W1 - W0
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten()
    Dc, W0c = D[:, cols], W0[:, cols]
    U, S, Vt = torch.linalg.svd(Dc, full_matrices=False)
    u = U[:, 0]
    a = u @ Dc                 # coefficient per column (Δw_j ≈ a_j u)
    p = u @ W0c                # old projection of column onto u
    ratio = a / p
    resid = (Dc - torch.outer(u, a)).norm() / Dc.norm()
    us[L] = u
    print(f"\n## L{L}: rank-1 residual {resid:.4f}")
    print(f"  a_j/p_j  mean={ratio.mean():.3f} std={ratio.std():.3f} min={ratio.min():.3f} max={ratio.max():.3f}")
    print(f"  corr(a, p) = {torch.corrcoef(torch.stack([a,p]))[0,1]:.4f}")
    print(f"  sign(a)==sign(p): {(torch.sign(a)==torch.sign(p)).float().mean():.2f}   |p_j|/||w0_j|| mean={(p.abs()/W0c.norm(dim=0)).mean():.3f}")
    # what fraction of ALL 3584 columns have large |uᵀw0|? were the 64 chosen as the top-|projection| ones?
    pall = u @ W0
    rank_of_chosen = (pall.abs().argsort(descending=True) == cols[:, None]).nonzero()[:, 1]
    print(f"  chosen cols' rank by |uᵀw0_j| among 3584: min={rank_of_chosen.min().item()} median={rank_of_chosen.float().median().item():.0f} max={rank_of_chosen.max().item()}")
    print(f"  |uᵀw0| chosen mean={pall[cols].abs().mean():.4f} vs all mean={pall.abs().mean():.4f}")
    # new projection
    pnew = u @ W1[:, cols]
    print(f"  new projection / old projection mean = {(pnew/p).mean():.3f}")
    out[L] = {"cols": cols.tolist(), "u": u.tolist(), "a": a.tolist(), "p_old": p.tolist(), "ratio": ratio.tolist()}
print(f"\ncos(u15, u19) = {torch.dot(us[15], us[19]):.4f}")
# unit-norm u vs residual-stream basis: how spread is u (is it a few dims?)
for L, u in us.items():
    top = u.abs().topk(5)
    print(f"L{L} u: top |dims| {top.indices.tolist()} values {[round(v,3) for v in top.values.tolist()]}, participation ratio {(u.abs().sum()**2/(u**2).sum()).item():.0f}/1024")
json.dump(out, open("notes/q1_rank1_structure_p50c.json", "w"))
