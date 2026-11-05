"""Generation eval with vLLM: one model dir, N GSM8K test problems, Astra's sampling, thinking on.
    python scripts/gen_eval_gsm8k.py <model_dir> <tag> [N=50] [max_tokens=16384]
Writes notes/gen_eval/<tag>.jsonl and prints acc / closed / cut / mean tokens / rep20."""
import os, sys, json, re, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

def rep_ratio(ids, n=20):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n)]; return 1 - len(set(g)) / max(1, len(g))
def extract(s):
    s = s.split("</think>")[-1]
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", s)
    if not m: return None
    x = m[-1].replace("\\$", "").replace("$", "").replace(",", "").replace("\\%", "").replace("%", "").replace("\\text{", "").replace("}", "").strip()
    x = re.sub(r"\\dfrac|\\frac", "", x)
    m2 = re.findall(r"-?\d+(?:\.\d+)?", x); return m2[-1] if m2 else x
def same(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except: return a == b

def main():
    model_dir, tag = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 50; MT = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    rows = [json.loads(l) for l in open("data/gsm8k_test.jsonl")][:N]
    llm = LLM(model=model_dir, gpu_memory_utilization=0.07, max_model_len=32768, enforce_eager=True, dtype="bfloat16", max_num_seqs=16)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"] + " Please reason step by step, and put your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=MT, seed=0)
    t = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t
    os.makedirs("notes/gen_eval", exist_ok=True)
    recs = []
    for r, o in zip(rows, outs):
        s = o.outputs[0].text; ids = list(o.outputs[0].token_ids); gold = r["answer"].split("####")[-1].strip().replace(",", "")
        pred = extract(s)
        recs.append(dict(id=r["id"], gold=gold, pred=pred, correct=bool(pred is not None and same(pred, gold)), n_tokens=len(ids),
                         closed="</think>" in s, cut=o.outputs[0].finish_reason == "length", rep20=rep_ratio(ids), text=s))
    with open(f"notes/gen_eval/{tag}.jsonl", "w") as f:
        for x in recs: f.write(json.dumps(x) + "\n")
    n = len(recs)
    summ = dict(tag=tag, n=n, acc=sum(x["correct"] for x in recs) / n, closed=sum(x["closed"] for x in recs) / n, cut=sum(x["cut"] for x in recs) / n,
                mean_tokens=sum(x["n_tokens"] for x in recs) / n, rep20=sum(x["rep20"] for x in recs) / n, sec=dt, tok_per_s=sum(x["n_tokens"] for x in recs) / dt)
    print("SUMMARY", json.dumps(summ))
    with open("notes/gen_eval/summary.jsonl", "a") as f: f.write(json.dumps(summ) + "\n")

if __name__ == "__main__":
    main()
