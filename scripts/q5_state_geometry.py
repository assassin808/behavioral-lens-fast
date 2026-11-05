"""Q5: does the edit move loop states toward productive states?  Per layer: linear probe (loop vs productive) fit on BASE states
(held-out texts), applied to base and edited states at loop positions; and centroid distances."""
import glob, gzip, json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
NT, MAXT = 8, 4096
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz"))[:4]:
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
print(len(texts), "texts", [round(float(m.mean()), 3) for _, _, m, _ in texts])
LAYERS = [12, 15, 16, 18, 19, 20, 22, 24]
H = {}
for tag in ["base", "p50c"]:
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/{tag}", dtype=torch.bfloat16).to("cuda").eval(); H[tag] = []
    with torch.no_grad():
        for tid, ids, m, P in texts:
            out = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True)
            H[tag].append({L: out.hidden_states[L][0].float().cpu().numpy() for L in LAYERS})
    del model; torch.cuda.empty_cache()
# subsample productive positions to balance
rng = np.random.default_rng(0)
print("hidden_states[L] | probe acc (base, held-out) | P(loop) base-loop | P(loop) edit-loop | P(loop) base-prod | P(loop) edit-prod | d(edit-loop, prod-centroid)/d(base-loop, prod-centroid)")
for L in LAYERS:
    pl_bl, pl_el, pl_bp, pl_ep, accs, dr = [], [], [], [], [], []
    for k in range(len(texts)):          # leave-one-text-out
        Xtr, ytr = [], []
        for j, (tid, ids, m, P) in enumerate(texts):
            if j == k: continue
            hb = H["base"][j][L]; lp = np.where(m)[0]; pp = np.where(~m & (np.arange(len(ids)) >= P))[0]
            pp = rng.choice(pp, min(len(pp), len(lp)), replace=False)
            Xtr += [hb[lp], hb[pp]]; ytr += [np.ones(len(lp)), np.zeros(len(pp))]
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1).fit((Xtr - mu) / sd, ytr)
        tid, ids, m, P = texts[k]; hb, he = H["base"][k][L], H["p50c"][k][L]
        lp = np.where(m)[0]; pp = np.where(~m & (np.arange(len(ids)) >= P))[0]
        f = lambda X: clf.predict_proba((X - mu) / sd)[:, 1]
        accs.append(0.5 * ((f(hb[lp]) > 0.5).mean() + (f(hb[pp]) < 0.5).mean()))
        pl_bl.append(f(hb[lp]).mean()); pl_el.append(f(he[lp]).mean()); pl_bp.append(f(hb[pp]).mean()); pl_ep.append(f(he[pp]).mean())
        cp = hb[pp].mean(0); dr.append(np.linalg.norm(he[lp] - cp, axis=1).mean() / np.linalg.norm(hb[lp] - cp, axis=1).mean())
    print(f"{L:3d} | {np.mean(accs):.3f} | {np.mean(pl_bl):.3f} | {np.mean(pl_el):.3f} | {np.mean(pl_bp):.3f} | {np.mean(pl_ep):.3f} | {np.mean(dr):.3f}")
