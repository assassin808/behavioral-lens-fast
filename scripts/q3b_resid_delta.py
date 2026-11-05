"""Q3b: residual-stream Δ (p50c − base) at loop vs productive positions, per layer; cos(Δ, u) and cos(Δ, -u) share."""
import glob, gzip, json, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
NT, MAXT = 6, 4096
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
base = load_file(f"{ROOT}/base/model.safetensors"); edit = load_file(f"{ROOT}/p50c/model.safetensors")
us = {}
for L in [15, 19]:
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"; D = edit[n].float() - base[n].float()
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten(); U, S, Vt = torch.linalg.svd(D[:, cols], full_matrices=False); us[L] = U[:, 0]
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
hs = {}
for tag in ["base", "p50c"]:
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/{tag}", dtype=torch.bfloat16).to("cuda").eval()
    hs[tag] = []
    with torch.no_grad():
        for tid, ids, m, P in texts:
            out = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True)
            hs[tag].append([h[0].float().cpu() for h in out.hidden_states])
    del model; torch.cuda.empty_cache()
print("layer-output  | rel ‖Δ‖ loop | rel ‖Δ‖ prod | cos(Δ,u15) loop/prod | cos(Δ,u19) loop/prod | share of Δ energy in span(u15,u19) loop/prod")
Q, _ = torch.linalg.qr(torch.stack([us[15], us[19]], 1))
for L in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
    rl, rp, c15l, c15p, c19l, c19p, el, ep = [], [], [], [], [], [], [], []
    for (tid, ids, m, P), hb, he in zip(texts, hs["base"], hs["p50c"]):
        d = he[L] - hb[L]; b = hb[L]
        ml = torch.tensor(m); mp = ~ml & (torch.arange(len(ids)) >= P)
        for mask, r, c15, c19, e in [(ml, rl, c15l, c19l, el), (mp, rp, c15p, c19p, ep)]:
            dd = d[mask]; r.append((dd.norm(dim=1) / b[mask].norm(dim=1)).mean().item())
            c15.append(torch.nn.functional.cosine_similarity(dd, us[15][None], dim=1).mean().item())
            c19.append(torch.nn.functional.cosine_similarity(dd, us[19][None], dim=1).mean().item())
            e.append((((dd @ Q) ** 2).sum(1) / (dd ** 2).sum(1)).mean().item())
    print(f"after L{L-1:2d}      | {np.mean(rl):.3f} | {np.mean(rp):.3f} | {np.mean(c15l):+.2f} / {np.mean(c15p):+.2f} | {np.mean(c19l):+.2f} / {np.mean(c19p):+.2f} | {np.mean(el):.2f} / {np.mean(ep):.2f}")
