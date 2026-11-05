"""Q3: who reads what the edit writes?  Teacher-force looping texts through base and p50c and measure, at loop positions:
 (a) per-layer change of attention / linear-attention / MLP outputs (‖Δ‖ / ‖base‖) downstream of L15;
 (b) copy attention at the full-attention layers 19 and 23: attention mass from a loop token to the earlier occurrence
     of the same n-gram (the token that was copied), base vs edited;
 (c) direct effect: does −u itself move the logits at loop positions (logit of the actual repeated token via the final norm)?
"""
import glob, gzip, json, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
qs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz"))[:3]:
    rows += [json.loads(l) for l in gzip.open(f, "rt")]
N = 32
def loop_info(ids):
    """for each position i: src[i] = index of the earlier token that the n-gram ending at i copies (i.e. previous occurrence's next-position), or -1"""
    first, src = {}, np.full(len(ids), -1)
    for i in range(N, len(ids)):
        g = tuple(ids[i - N:i])
        if g in first: src[i] = first[g]          # position right after the earlier copy of the n-gram = the token being copied
        else: first[g] = i
    return src
texts = []
for r in rows:
    if "</think>" in r["text"]: continue
    q = qs[r["id"]] + " Please reason step by step, and put your final answer within \\boxed{}."
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; gid = tok(r["text"], add_special_tokens=False).input_ids
    ids = (pid + gid)[:MAXT]; src = loop_info(ids); src[:len(pid)] = -1
    if (src >= 0).mean() > 0.03: texts.append((r["id"], ids, src, len(pid)))
    if len(texts) >= NT: break
print(f"{len(texts)} looping texts, loop_frac={[round(float((s>=0).mean()),3) for _,_,s,_ in texts]}")

FULL = [3, 7, 11, 15, 19, 23]
def run(tag):
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/{tag}", dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    layers = model.model.layers
    store = {}
    def hook(name):
        def h(mod, inp, out):
            store[name] = (out[0] if isinstance(out, tuple) else out)[0].detach().float()
        return h
    hs = []
    for L, lay in enumerate(layers):
        hs.append(lay.mlp.register_forward_hook(hook(f"mlp{L}")))
        att = getattr(lay, "self_attn", None) or getattr(lay, "linear_attn")
        hs.append(att.register_forward_hook(hook(f"attn{L}")))
    res = []
    with torch.no_grad():
        for tid, ids, src, P in texts:
            store.clear()
            out = model.model(torch.tensor([ids]).cuda(), output_attentions=True, output_hidden_states=True)
            lp = np.where(src >= 0)[0]; pp = np.where((src < 0) & (np.arange(len(ids)) >= P))[0]
            rec = {"id": tid, "layers": {}, "copy_attn": {}, "resid": {}}
            for L in range(24):
                for k in ("attn", "mlp"):
                    x = store[f"{k}{L}"]; rec["layers"][f"{k}{L}"] = x[lp].cpu()   # keep loop-position outputs for Δ
            # copy attention: attn weights [heads, T, T]; for loop pos i, mass on src[i] (and on src[i]-1..src[i]+1 window)
            atts = out.attentions
            for L in FULL:
                A = atts[FULL.index(L)] if len(atts) == len(FULL) else atts[L]
                if A is None: continue
                A = A[0]                                   # [H, T, T]
                i = torch.tensor(lp).cuda(); j = torch.tensor(src[lp]).cuda()
                m_exact = A[:, i, j].mean(1)               # per head, mean over loop positions
                m_win = sum(A[:, i, (j + d).clamp(0, len(ids) - 1)] for d in (-1, 0, 1)).mean(1)
                # for productive positions: attention mass to the max-attended earlier position (not defined by copy) -> skip; use entropy instead
                rec["copy_attn"][L] = dict(exact=m_exact.cpu().tolist(), win3=m_win.cpu().tolist())
            for L in [15, 16, 19, 20, 23, 24]:
                rec["resid"][L] = out.hidden_states[L][0][lp].cpu()
            res.append(rec)
            del out
    for h in hs: h.remove()
    del model; torch.cuda.empty_cache()
    return res

rb, re_ = run("base"), run("p50c")
print("\n### (a) per-layer relative change of component outputs at loop positions, ‖edited − base‖ / ‖base‖ (mean over texts)")
for L in range(15, 24):
    s = []
    for k in ("attn", "mlp"):
        v = np.mean([((b["layers"][f"{k}{L}"] - e["layers"][f"{k}{L}"]).norm(dim=1) / b["layers"][f"{k}{L}"].norm(dim=1)).mean().item() for b, e in zip(rb, re_)])
        s.append(f"{k}{L}{'*' if L in FULL and k=='attn' else ''}={v:.3f}")
    print("  ", "  ".join(s))
print("\n### (b) copy attention (mass on the token being copied), per head, base → edited")
for L in [19, 23]:
    for key in ("exact", "win3"):
        b = np.mean([r["copy_attn"][L][key] for r in rb], 0); e = np.mean([r["copy_attn"][L][key] for r in re_], 0)
        print(f"  L{L} {key}: base {np.round(b,3).tolist()}\n           edit {np.round(e,3).tolist()}   (sum over heads {b.sum():.3f} → {e.sum():.3f})")
print("\n### (c) residual-stream change at loop positions: ‖Δ‖/‖base‖ and cos(Δ, direction of the edit's direct write)")
for L in [15, 16, 19, 20, 23, 24]:
    v = np.mean([((b["resid"][L] - e["resid"][L]).norm(dim=1) / b["resid"][L].norm(dim=1)).mean().item() for b, e in zip(rb, re_)])
    print(f"  after layer {L-1 if L>0 else 'emb'} (hidden_states[{L}]): {v:.3f}")
torch.save({"base": rb, "edit": re_}, "notes/q3_who_reads_u.pt")
