"""Test 7d: inference-time steering with u (constant −c·u added to the L15 and L19 MLP outputs), HF batched generation,
Astra's prompt/sampling (presence penalty implemented as a logits processor), Astra's grader.
    python scripts/steer_eval.py <c values comma-separated, e.g. 0,0.5,1,2> [--n 100] [--max-tokens 8192] [--batch 32] [--model base|p50c]"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, "scripts")
from astra_eval import build_prompt, grade, loop_signals
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"

class PresencePenalty(LogitsProcessor):
    def __init__(self, p, prompt_len): self.p, self.P = p, prompt_len
    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.P:]
        if gen.shape[1] == 0: return scores
        mask = torch.zeros_like(scores, dtype=torch.bool); mask.scatter_(1, gen, True)
        return scores - self.p * mask.float()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cs"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--model", default="base"); ap.add_argument("--sets", default="gsm8k")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/base"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/{a.model}", dtype=torch.bfloat16).to("cuda").eval()
    u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")).cuda().to(torch.bfloat16) for L in [15, 19]}
    state = {"c": 0.0}
    def mk(L):
        def h(mod, inp, out): return out - state["c"] * u[L] if state["c"] else out
        return h
    for L in [15, 19]: model.model.layers[L].mlp.down_proj.register_forward_hook(mk(L))
    rows = []
    for name in a.sets.split(","):
        if name == "gsm8k": rows += [dict(set="gsm8k", id=r["id"], problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for r in map(json.loads, open("data/gsm8k_test.jsonl"))][:a.n]
        else: rows += [dict(set="math500", **{k: r[k] for k in ("id", "problem", "answer")}) for r in map(json.loads, open("data/math500.jsonl"))][:a.n]
    os.makedirs("notes/steer", exist_ok=True)
    for c in [float(x) for x in a.cs.split(",")]:
        state["c"] = c; tag = f"{a.model}_steer{c:g}"; recs = []; t0 = time.time(); ntok = 0
        for i in range(0, len(rows), a.batch):
            B = rows[i:i + a.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r["problem"])}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in B]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda"); P = enc.input_ids.shape[1]
            torch.manual_seed(0)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                     logits_processor=LogitsProcessorList([PresencePenalty(1.5, P)]), pad_token_id=tok.pad_token_id)
            for r, o in zip(B, out):
                g = o[P:].tolist(); g = [t for t in g if t != tok.pad_token_id]; text = tok.decode(g, skip_special_tokens=True); ntok += len(g)
                ok, pred = grade(text, r["answer"]); ls = loop_signals(text)
                recs.append(dict(set=r["set"], id=r["id"], gold=r["answer"], pred=pred, correct=ok, n_tokens=len(g), cut=len(g) >= a.max_tokens, closed="</think>" in text,
                                 loop_end=bool(ls["verbatim_loop_end"] or ls["repetitive_end"]), text=text))
            print(f"  {tag}: {i+len(B)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        with open(f"notes/steer/{tag}.jsonl", "w") as f:
            for x in recs: f.write(json.dumps(x) + "\n")
        done = [x for x in recs if not x["cut"]]
        summ = dict(tag=tag, c=c, n=len(recs), completion=len(done)/len(recs), acc_all=sum(x["correct"] for x in recs)/len(recs), acc_on_completed=sum(x["correct"] for x in done)/max(1,len(done)),
                    loop_end=sum(x["loop_end"] for x in recs)/len(recs), mean_tokens=ntok/len(recs), sec=round(time.time()-t0), tok_per_s=round(ntok/(time.time()-t0)))
        print("SUMMARY", json.dumps(summ), flush=True)
        with open("notes/steer/summary.jsonl", "a") as f: f.write(json.dumps(summ) + "\n")

if __name__ == "__main__":
    main()
