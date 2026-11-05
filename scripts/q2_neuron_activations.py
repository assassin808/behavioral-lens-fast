"""Q2: when do the 128 edited neurons fire?  Teacher-force real unedited generations (looping and productive) and
compare activations in looping vs non-looping token positions; signed write along u before/after the edit.

    python scripts/q2_neuron_activations.py [n_texts_per_class] [max_tokens]
"""
import glob, gzip, json, sys, os
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/base", dtype=torch.bfloat16).to("cuda").eval()
base = load_file(f"{ROOT}/base/model.safetensors"); edit = load_file(f"{ROOT}/p50c/model.safetensors")

# edit geometry (from Q1)
geo = {}
for L in [15, 19]:
    n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    W0, W1 = base[n].float(), edit[n].float(); D = W1 - W0
    cols = torch.nonzero(D.abs().sum(0) > 0).flatten()
    U, S, Vt = torch.linalg.svd(D[:, cols], full_matrices=False); u = U[:, 0]
    if (u @ D[:, cols]).mean() * (u @ W0[:, cols]).mean() > 0: pass
    geo[L] = dict(cols=cols, u=u.cuda(), p_old=(u @ W0).cuda(), p_new=(u @ W1).cuda(), W0=W0.cuda(), W1=W1.cuda())

# GSM8K questions
qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
print("gsm8k questions loaded", len(qs))

rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz"))[:3]:
    rows += [json.loads(l) for l in gzip.open(f, "rt")]
looping = [r for r in rows if "</think>" not in r["text"]][:NT]
productive = [r for r in rows if "</think>" in r["text"]][:NT]
print(f"{len(looping)} looping, {len(productive)} productive texts")

def loop_mask(ids, n=32):
    """token i is 'in loop' if the n-gram ending at i already occurred earlier."""
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m

acts = {15: [], 19: []}
def mk_hook(L):
    def h(mod, inp, out): acts[L].append(inp[0][0].detach())
    return h
layers = model.model.layers
hs = [layers[L].mlp.down_proj.register_forward_hook(mk_hook(L)) for L in [15, 19]]

summary = {15: [], 19: []}
per_text = []
with torch.no_grad():
    for cls, group in [("loop", looping), ("prod", productive)]:
        for r in group:
            q = qs.get(r["id"], "Solve the problem.") + " Please reason step by step, and put your final answer within \\boxed{}."
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            pid = tok(prompt).input_ids; gid = tok(r["text"], add_special_tokens=False).input_ids
            ids = (pid + gid)[:MAXT]; P = len(pid)
            m = loop_mask(ids); m[:P] = False
            for L in acts: acts[L].clear()
            out = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True)
            rec = dict(id=r["id"], cls=cls, T=len(ids), loop_frac=float(m[P:].mean()))
            for L in [15, 19]:
                a = acts[L][0].float()                           # [T, 3584] activations (input to down_proj)
                g = geo[L]; c = g["cols"]
                write_u_old = a @ g["p_old"]; write_u_new = a @ g["p_new"]  # [T] MLP output along u (all neurons)
                resid = out.hidden_states[L + 1][0].float() @ g["u"]          # residual after layer L along u
                sel = a[:, c]
                mp, ml = (~m) & (np.arange(len(ids)) >= P), m
                mp_t, ml_t = torch.tensor(mp).cuda(), torch.tensor(ml).cuda()
                def mean(x, mask): return x[mask].mean().item() if mask.sum() > 0 else float("nan")
                rec[f"L{L}"] = dict(
                    sel_act_mean_prod=mean(sel, mp_t), sel_act_mean_loop=mean(sel, ml_t),
                    sel_abs_act_mean_prod=mean(sel.abs(), mp_t), sel_abs_act_mean_loop=mean(sel.abs(), ml_t),
                    all_abs_act_mean_prod=mean(a.abs(), mp_t), all_abs_act_mean_loop=mean(a.abs(), ml_t),
                    write_u_old_prod=mean(write_u_old, mp_t), write_u_old_loop=mean(write_u_old, ml_t),
                    write_u_new_prod=mean(write_u_new, mp_t), write_u_new_loop=mean(write_u_new, ml_t),
                    sel_write_u_old_prod=mean(sel @ g["p_old"][c], mp_t), sel_write_u_old_loop=mean(sel @ g["p_old"][c], ml_t),
                    resid_u_prod=mean(resid, mp_t), resid_u_loop=mean(resid, ml_t),
                )
                # per-neuron loop selectivity for all 3584 neurons: d' = (mean_loop - mean_prod)/pooled std of |act|
                if ml.sum() > 50 and mp.sum() > 50:
                    d = (a.abs()[ml_t].mean(0) - a.abs()[mp_t].mean(0)) / (a.abs().std(0) + 1e-6)
                    summary[L].append(d.cpu())
            per_text.append(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith("L")}),
                  {L: {k: round(v, 4) for k, v in rec[f"L{L}"].items() if "write_u" in k or "resid" in k} for L in [15, 19]})

print("\n### per-neuron loop selectivity (d' of |act|, loop vs productive positions), averaged over looping texts")
for L in [15, 19]:
    if not summary[L]: continue
    d = torch.stack(summary[L]).mean(0); c = geo[L]["cols"].cpu()
    rank = d.argsort(descending=True); pos = torch.tensor([(rank == j).nonzero().item() for j in c])
    print(f"L{L}: selected-64 d' mean={d[c].mean():.3f} (others {d.mean():.3f}); rank of selected among 3584: median {pos.float().median():.0f}, "
          f"#in top-64 {(pos < 64).sum().item()}, #in top-256 {(pos < 256).sum().item()}")
    print(f"   top-10 loop-selective neurons overall: {rank[:10].tolist()}  (selected? {[int(j in set(c.tolist())) for j in rank[:10].tolist()]})")
json.dump(dict(per_text=per_text, cols={L: geo[L]["cols"].tolist() for L in geo},
               dprime={L: torch.stack(summary[L]).mean(0).tolist() for L in summary if summary[L]}),
          open("notes/q2_neuron_activations.json", "w"))
