"""Pilot: model structure, u check vs edit diff, fla speed, loop masks, eyeball samples, public J-lens read of u."""
import time, sys, os, numpy as np, torch
from test24_common import *
m, tok = load_model("base")
lm, layers, norm, head = lm_parts(m)
print(type(m).__name__, len(layers), [type(l.linear_attn if hasattr(l,'linear_attn') else l.self_attn).__name__ for l in layers][:4])
print("layer types", [getattr(l, "layer_type", "?") for l in layers])
from safetensors.torch import load_file
import glob
b = load_file(sorted(glob.glob(f"{MODEL_BASE}/*.safetensors"))[0]); e = load_file(sorted(glob.glob(f"{MODEL_P50C}/*.safetensors"))[0])
U = get_u()
for L in [15, 19]:
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"; D = e[n].float() - b[n].float()
    Uu, S, _ = torch.linalg.svd(D, full_matrices=False)
    print(L, "svd", S[:3].tolist(), "cos(U1,u)", float(Uu[:, 0] @ torch.tensor(U[L])))
print("changed tensors", [k for k in b if not torch.equal(b[k], e[k])])
del b, e
items = chat_items(tok, maxt=16384)
print("n items", len(items), "tokens", sum(len(x["ids"]) for x in items) / 1e6, "M; loop frac", np.mean(np.concatenate([x["loop"] for x in items])))
for tag in ["unedited", "p50c"]:
    for ds in DSETS:
        s = [x for x in items if x["gen_by"] == tag and x["ds"] == ds]
        print(tag, ds, len(s), "tok/item", int(np.mean([len(x["ids"]) for x in s])), "loopfrac", round(float(np.mean([x["loop"].mean() for x in s])), 3), "finished", np.mean([x["finished"] for x in s]))
# eyeball: 20 loop snippets
random.seed(0)
for x in random.sample([x for x in items if x["loop"].mean() > 0.05], 10):
    i = int(np.flatnonzero(x["loop"])[len(np.flatnonzero(x["loop"])) // 2])
    print("----", x["id"], x["gen_by"], "loopfrac", round(x["loop"].mean(), 3), "pos", i, repr(tok.decode(x["ids"][i - 80:i])), "|| NEXT", repr(tok.decode(x["ids"][i:i + 5])))
# speed
ids = torch.tensor([items[0]["ids"][:8192]]).cuda()
for T in [2048, 8192]:
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): o = m.model(ids[:, :T], output_hidden_states=True)
    torch.cuda.synchronize(); print("T", T, "sec", round(time.time() - t, 2), "n hs", len(o.hidden_states), "mem GB", torch.cuda.max_memory_allocated() / 1e9)
hs = o.hidden_states
print("norms per layer", [round(float(h[0].float().norm(dim=-1).median()), 1) for h in hs])
# check hidden_states[24] vs normed: is the last one post-norm?
print("last hs == norm(prev)?", float((norm(hs[23]) - hs[24]).abs().max()))
import jlens
jm = jlens.from_hf(m, tok)
for nm, p in [("jlens", f"{DATA}/ext/jlens/Qwen3.5-0.8B_jacobian_lens.pt"), ("jpp", f"{DATA}/ext/jpp/Qwen3.5-0.8B_jpp_lens.pt")]:
    lens = jlens.JacobianLens.load(p); print(nm, lens)
    for L in range(15, 23):
        for sgn in [1, -1]:
            v = torch.tensor(U[15] * sgn).cuda()[None]
            lg = jm.unembed(lens.transport(v.float(), L).to(torch.bfloat16))[0].float()
            print(nm, L, "+u" if sgn > 0 else "-u", [tok.decode([t]) for t in lg.topk(12).indices.tolist()])
