"""Build variant checkpoints (same rule family as Q4) as full model dirs for vLLM."""
import os, shutil, sys, torch, numpy as np
from safetensors.torch import load_file, save_file
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
base = load_file(f"{ROOT}/base/model.safetensors"); edit = load_file(f"{ROOT}/p50c/model.safetensors")
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"
geo = {}
for L in [15, 19]:
    W0, W1 = base[NAME.format(L)].float(), edit[NAME.format(L)].float(); D = W1 - W0
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten(); U, S, Vt = torch.linalg.svd(D[:, cols], full_matrices=False)
    geo[L] = dict(W0=W0, W1=W1, cols=cols, u=U[:, 0])
torch.manual_seed(0)
def variant(name, L):
    g = geo[L]; W0, u, cols = g["W0"], g["u"], g["cols"]
    if name.startswith("sel_alpha"):
        a = float(name[9:]); W = W0.clone(); W[:, cols] = W0[:, cols] - a * torch.outer(u, u @ W0[:, cols])
    elif name.startswith("all_alpha"):
        a = float(name[9:]); W = W0 - a * torch.outer(u, u @ W0)
    elif name == "rand_cols":
        rc = torch.randperm(W0.shape[1])[:64]; W = W0.clone(); W[:, rc] = W0[:, rc] - 6 * torch.outer(u, u @ W0[:, rc])
    elif name.startswith("top64_"):                 # top64_<criterion>_alpha<a>: our re-ranked columns, same rule
        crit, a = name[6:].rsplit("_alpha", 1); c = torch.tensor(np.load(f"shared/top64_{crit}_L{L}.npy")); a = float(a)
        W = W0.clone(); W[:, c] = W0[:, c] - a * torch.outer(u, u @ W0[:, c])
    elif name.startswith("rand_dir") and name != "rand_dir":   # rand_dirNN: random unit direction, Astra cols, α=6 (same generator order as q4b)
        k = int(name[8:]); gen = torch.Generator().manual_seed(0); r = None
        for i in range(k + 1):
            r = {LL: torch.randn(1024, generator=gen) for LL in [15, 19]}
        r = r[L] / r[L].norm(); W = W0.clone(); W[:, cols] = W0[:, cols] - 6 * torch.outer(r, r @ W0[:, cols])
    elif name == "rand_dir":
        r = torch.randn_like(u); r /= r.norm(); W = W0.clone(); d = (g["W1"] - W0)[:, cols].norm(dim=0)
        W[:, cols] = W0[:, cols] + torch.outer(r, d) * torch.sign(u @ W0[:, cols])
    else: raise ValueError(name)
    return W.to(torch.bfloat16)
for name in sys.argv[1:]:
    out = f"{ROOT}/variants/{name}"; os.makedirs(out, exist_ok=True)
    for f in os.listdir(f"{ROOT}/base"):
        if f != "model.safetensors": shutil.copy(f"{ROOT}/base/{f}", out)
    t = dict(base)
    for L in [15, 19]: t[NAME.format(L)] = variant(name, L).contiguous()
    save_file(t, f"{out}/model.safetensors", metadata={"format": "pt"}); print("built", out)
