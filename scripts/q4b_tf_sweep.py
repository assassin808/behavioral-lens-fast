"""Test 6 (teacher-forced): α sweep on Astra's columns, true top-64 variants, ≥20 random directions, random columns.
Same NLL-at-loop/productive-positions metric as Q4 (6 looping GSM8K generations, 4096 tokens)."""
import glob, gzip, json, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
NT, MAXT, NRAND = 6, 4096, 20
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/base", dtype=torch.bfloat16).to("cuda").eval()
base = load_file(f"{ROOT}/base/model.safetensors"); layers = model.model.layers
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"
W0 = {L: base[NAME.format(L)].float() for L in [15, 19]}
u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")) for L in [15, 19]}
cols = {L: {"astra": np.load(f"shared/astra_cols_L{L}.npy"), "top64_dprime": np.load(f"shared/top64_by_dprime_L{L}.npy"),
            "top64_uproj": np.load(f"shared/top64_by_uproj_L{L}.npy"), "top64_combined": np.load(f"shared/top64_combined_L{L}.npy")} for L in [15, 19]}
def edited(L, c, direction, alpha):
    W = W0[L].clone(); c = torch.tensor(c); W[:, c] = W0[L][:, c] - alpha * torch.outer(direction, direction @ W0[L][:, c]); return W
def set_w(ws):
    for L, W in ws.items(): layers[L].mlp.down_proj.weight.data.copy_(W.to(torch.bfloat16))
qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz"))[:3]: rows += [json.loads(l) for l in gzip.open(f, "rt")]
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
texts = []
for r in rows:
    if "</think>" in r["text"]: continue
    prompt = tok.apply_chat_template([{"role": "user", "content": qs[r["id"]] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; ids = (pid + tok(r["text"], add_special_tokens=False).input_ids)[:MAXT]; m = loop_mask(ids); m[:len(pid)] = False
    if m.mean() > 0.03: texts.append((r["id"], ids, m, len(pid)))
    if len(texts) >= NT: break
@torch.no_grad()
def nll_at(ids, positions):
    h = model.model(torch.tensor([ids]).cuda()).last_hidden_state[0]; tgt = torch.tensor(ids[1:]).cuda(); out = []
    for s in range(0, len(positions), 512):
        p = torch.tensor(positions[s:s+512]).cuda(); out.append(torch.nn.functional.cross_entropy(model.lm_head(h[p]).float(), tgt[p], reduction="none"))
    return torch.cat(out)
def evaluate(ws):
    set_w(ws); lo, pr = [], []
    for tid, ids, m, P in texts:
        pos = np.arange(len(ids) - 1); lp = pos[m[1:] & (pos >= P)]; pp = pos[(~m[1:]) & (pos >= P)]
        lo.append(nll_at(ids, lp.tolist()).mean().item()); pr.append(nll_at(ids, pp.tolist()).mean().item())
    return float(np.mean(lo)), float(np.mean(pr))
variants = [("base", {L: W0[L] for L in W0})]
for a in [1, 2, 4, 6, 8]: variants.append((f"astra_alpha{a}", {L: edited(L, cols[L]["astra"], u[L], a) for L in W0}))
for which in ["top64_dprime", "top64_uproj", "top64_combined"]:
    for a in [6]: variants.append((f"{which}_alpha{a}", {L: edited(L, cols[L][which], u[L], a) for L in W0}))
g = torch.Generator().manual_seed(0)
for k in range(NRAND):
    r = {L: torch.randn(1024, generator=g) for L in W0}; r = {L: v / v.norm() for L, v in r.items()}
    variants.append((f"rand_dir{k:02d}_alpha6", {L: edited(L, cols[L]["astra"], r[L], 6) for L in W0}))
for k in range(5):
    rc = {L: torch.randperm(3584, generator=g)[:64].numpy() for L in W0}
    variants.append((f"rand_cols{k}_alpha6", {L: edited(L, rc[L], u[L], 6) for L in W0}))
res = {}
for name, ws in variants:
    lo, pr = evaluate(ws); res[name] = dict(loop=lo, prod=pr)
    b = res["base"]; print(f"{name:26s} loop {lo:.3f} ({lo-b['loop']:+.3f})  prod {pr:.3f} ({pr-b['prod']:+.3f})", flush=True)
rd = [v for k, v in res.items() if k.startswith("rand_dir")]
print(f"\nrandom directions (n={len(rd)}): Δloop mean {np.mean([v['loop']-res['base']['loop'] for v in rd]):+.4f} max {np.max([v['loop']-res['base']['loop'] for v in rd]):+.4f}; Astra α6 Δloop {res['astra_alpha6']['loop']-res['base']['loop']:+.4f}")
json.dump(res, open("notes/q4b_tf_sweep.json", "w"), indent=1)
