"""Q6 (teacher-forced): does the anti-loop effect transfer to non-math generations (IFEval, MMLU-Pro) and how much does the edit
perturb non-loop text there?  Same NLL-at-loop/productive-positions metric as Q4, base vs p50c, no prompt (generic header)."""
import glob, gzip, json, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MAXT, NT = 4096, 8
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
header = tok.apply_chat_template([{"role": "user", "content": "Answer the question."}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
pid = tok(header).input_ids
sets = {}
for ds in ["ifeval", "mmlupro1k", "gpqa", "math500"]:
    rows = []
    for f in sorted(glob.glob(f"astra_logs/exp8_math/eval/unedited/parts/{ds}_s*/{ds}_first_samples.jsonl.gz"))[:4]:
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
    loop, prod = [], []
    for r in rows:
        ids = (pid + tok(r["text"], add_special_tokens=False).input_ids)[:MAXT]; m = loop_mask(ids); m[:len(pid)] = False
        (loop if m.mean() > 0.03 else prod).append((r["id"], ids, m, len(pid)))
    sets[ds] = dict(loop=loop[:NT], prod=prod[:NT], n_rows=len(rows), n_loop=len(loop), n_closed=sum("</think>" in r["text"] for r in rows))
    print(f"{ds}: {len(rows)} samples, {len(loop)} with loops (>3%), {sets[ds]['n_closed']} closed </think>")
models = {}
res = {}
for tag in ["base", "p50c"]:
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/{tag}", dtype=torch.bfloat16).to("cuda").eval()
    @torch.no_grad()
    def nll_at(ids, positions):
        h = model.model(torch.tensor([ids]).cuda()).last_hidden_state[0]; tgt = torch.tensor(ids[1:]).cuda(); out = []
        for s in range(0, len(positions), 512):
            p = torch.tensor(positions[s:s+512]).cuda(); out.append(torch.nn.functional.cross_entropy(model.lm_head(h[p]).float(), tgt[p], reduction="none"))
        return torch.cat(out)
    for ds, S in sets.items():
        for cls in ["loop", "prod"]:
            for tid, ids, m, P in S[cls]:
                pos = np.arange(len(ids) - 1); lp = pos[m[1:] & (pos >= P)]; pp = pos[(~m[1:]) & (pos >= P)]
                r = res.setdefault((ds, cls, tag), {"loop": [], "prod": []})
                if len(lp) > 20: r["loop"].append(nll_at(ids, lp.tolist()).mean().item())
                r["prod"].append(nll_at(ids, pp.tolist()).mean().item())
    del model; torch.cuda.empty_cache()
print("\ndataset | texts | NLL loop-pos base→p50c | NLL prod-pos base→p50c")
for ds in sets:
    for cls in ["loop", "prod"]:
        b, e = res.get((ds, cls, "base")), res.get((ds, cls, "p50c"))
        if not b or not b["prod"]: continue
        lb = f"{np.mean(b['loop']):.3f} → {np.mean(e['loop']):.3f}" if b["loop"] else "—"
        print(f"{ds} {cls}-texts | {len(b['prod'])} | {lb} | {np.mean(b['prod']):.3f} → {np.mean(e['prod']):.3f}")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, open("notes/q6_tf_nonmath.json", "w"), indent=1)
