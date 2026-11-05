"""Q4 (teacher-forced): does an intervention lower the probability of the repeated continuation inside loops while leaving
productive positions alone?  Compare: base / p50c / full-column rank-1 reflection W−α u(uᵀW) / constant bias −c·u at the
MLP output / norm-matched random direction (same 64 columns).  Metric: NLL of the actual next token at loop vs productive
positions, and P(next token) ratio.

    python scripts/q4_tf_interventions.py [n_loop_texts] [max_tokens]
"""
import glob, gzip, json, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/base", dtype=torch.bfloat16).to("cuda").eval()
base = load_file(f"{ROOT}/base/model.safetensors"); edit = load_file(f"{ROOT}/p50c/model.safetensors")
layers = model.model.layers
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"
geo = {}
for L in [15, 19]:
    W0, W1 = base[NAME.format(L)].float(), edit[NAME.format(L)].float(); D = W1 - W0
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten()
    U, S, Vt = torch.linalg.svd(D[:, cols], full_matrices=False)
    geo[L] = dict(W0=W0, W1=W1, cols=cols, u=U[:, 0])

torch.manual_seed(0)
def variant_weights(name):
    out = {}
    for L, g in geo.items():
        W0, u, cols = g["W0"], g["u"], g["cols"]
        if name == "base": W = W0
        elif name == "p50c": W = g["W1"]
        elif name.startswith("sel_alpha"):            # same 64 columns, different α
            a = float(name.split("alpha")[1]); W = W0.clone(); W[:, cols] = W0[:, cols] - a * torch.outer(u, u @ W0[:, cols])
        elif name.startswith("all_alpha"):            # all 3584 columns (p10c-style), α
            a = float(name.split("alpha")[1]); W = W0 - a * torch.outer(u, u @ W0)
        elif name == "rand_dir":                      # random unit direction r, same 64 cols, same ‖Δ‖ per column
            r = torch.randn_like(u); r /= r.norm(); W = W0.clone()
            d = (g["W1"] - W0)[:, cols].norm(dim=0); W[:, cols] = W0[:, cols] + torch.outer(r, d) * torch.sign(u @ W0[:, cols])
        elif name == "rand_cols":                     # same rule (−6 u uᵀ), 64 random other columns
            rc = torch.randperm(W0.shape[1])[:64]; W = W0.clone(); W[:, rc] = W0[:, rc] - 6 * torch.outer(u, u @ W0[:, rc])
        elif name == "sel_zero":                      # ablate the 64 neurons' write along u (α=1) handled by sel_alpha1
            raise ValueError
        else: raise ValueError(name)
        out[L] = W
    return out

def set_weights(ws):
    for L, W in ws.items():
        layers[L].mlp.down_proj.weight.data.copy_(W.to(torch.bfloat16))

bias = {}
def mk_bias_hook(L):
    def h(mod, inp, out):
        if L in bias: return out + bias[L].to(out.dtype)
    return h
for L in [15, 19]: layers[L].mlp.down_proj.register_forward_hook(mk_bias_hook(L))

qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz"))[:3]:
    rows += [json.loads(l) for l in gzip.open(f, "rt")]
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
texts = []
for r in rows:
    if "</think>" in r["text"]: continue
    q = qs[r["id"]] + " Please reason step by step, and put your final answer within \\boxed{}."
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; gid = tok(r["text"], add_special_tokens=False).input_ids
    ids = (pid + gid)[:MAXT]; m = loop_mask(ids); m[:len(pid)] = False
    if m.mean() > 0.03: texts.append((r["id"], ids, m, len(pid)))
    if len(texts) >= NT: break
print(f"{len(texts)} looping texts, loop_frac={[round(float(m.mean()),3) for _,_,m,_ in texts]}")

lm_head = model.lm_head
@torch.no_grad()
def nll_at(ids, positions):
    out = model.model(torch.tensor([ids]).cuda())
    h = out.last_hidden_state[0]                      # [T, d]
    tgt = torch.tensor(ids[1:]).cuda()
    res = []
    for s in range(0, len(positions), 512):
        p = torch.tensor(positions[s:s+512]).cuda()
        lg = lm_head(h[p]).float()
        res.append(torch.nn.functional.cross_entropy(lg, tgt[p], reduction="none"))
    return torch.cat(res)

variants = ["base", "p50c", "sel_alpha1", "sel_alpha2", "sel_alpha6", "sel_alpha10", "all_alpha1", "all_alpha1.5", "all_alpha2",
            "rand_dir", "rand_cols", "bias-0.5", "bias-1", "bias-2"]
results = {v: {"loop": [], "prod": []} for v in variants}
for v in variants:
    bias.clear()
    if v.startswith("bias"):
        set_weights(variant_weights("base")); c = float(v[4:])
        for L in [15, 19]: bias[L] = c * geo[L]["u"].cuda()
    else:
        set_weights(variant_weights(v))
    for tid, ids, m, P in texts:
        pos = np.arange(len(ids) - 1)
        lp = pos[m[1:] & (pos >= P)]; pp = pos[(~m[1:]) & (pos >= P)]
        nl = nll_at(ids, lp.tolist()); npd = nll_at(ids, pp.tolist())
        results[v]["loop"].append(nl.mean().item()); results[v]["prod"].append(npd.mean().item())
    print(f"{v:14s} NLL loop {np.mean(results[v]['loop']):.3f}  NLL prod {np.mean(results[v]['prod']):.3f}   "
          f"(Δ vs base: loop {np.mean(results[v]['loop'])-np.mean(results['base']['loop']):+.3f}, prod {np.mean(results[v]['prod'])-np.mean(results['base']['prod']):+.3f})", flush=True)
json.dump(results, open("notes/q4_tf_interventions.json", "w"), indent=1)
