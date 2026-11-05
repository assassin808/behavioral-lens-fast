"""Base model, thinking OFF, on the same GSM8K / MATH-500 problems Astra logged first samples for (200 + 200), Astra's sampling
(think35: T 0.6, top-p 0.95, top-k 20, presence 1.5), 16k cap, 1 sample. Used by test 8 (does u light up when no-think loops?) and the
thinking-on vs -off comparison along u.  Out: $T38_DATA/gen_base_nothink.jsonl (gitignored) + one summary line printed.

    python scripts/test8_gen_nothink.py [model_dir] [tag]
"""
import os, sys, json, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODELS, DATA_OUT, astra_rows, question, loop_mask

model_dir = sys.argv[1] if len(sys.argv) > 1 else f"{MODELS}/base"
tag = sys.argv[2] if len(sys.argv) > 2 else "base_nothink"
from vllm import LLM, SamplingParams
llm = LLM(model=model_dir, gpu_memory_utilization=0.85, max_model_len=20480, dtype="bfloat16", max_num_seqs=256, seed=0)
tok = llm.get_tokenizer()
rows = astra_rows("unedited", "gsm8k") + astra_rows("unedited", "math500")
prompts = [tok.apply_chat_template([{"role": "user", "content": question(r["id"])}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for r in rows]
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=16384, seed=0))
try:
    from astra_eval import grade
except Exception:
    grade = None
os.makedirs(DATA_OUT, exist_ok=True)
recs = []
for r, o in zip(rows, outs):
    s = o.outputs[0]; ids = list(s.token_ids); m = loop_mask(ids)
    ok = grade(s.text, r["answer"])[0] if (grade and r["answer"] is not None) else None
    recs.append(dict(id=r["id"], src=r["src"], answer=r["answer"], text=s.text, n_tokens=len(ids), cut=s.finish_reason == "length",
                     loop_frac=float(m.mean()) if len(m) else 0.0, correct=ok))
with open(f"{DATA_OUT}/gen_{tag}.jsonl", "w") as f:
    for x in recs: f.write(json.dumps(x) + "\n")
summ = dict(tag=tag, sec=round(time.time() - t0))
for src in ["gsm8k", "math"]:
    R = [x for x in recs if x["src"] == src]
    summ[src] = dict(n=len(R), cut=sum(x["cut"] for x in R) / len(R), loop_frac_gt_5pct=sum(x["loop_frac"] > .05 for x in R) / len(R),
                     mean_tokens=sum(x["n_tokens"] for x in R) / len(R),
                     acc=(sum(bool(x["correct"]) for x in R) / len(R)) if grade else None)
print("SUMMARY", json.dumps(summ), flush=True)
json.dump(summ, open(f"{REPO_SUM}" if (REPO_SUM := os.environ.get("SUMMARY_OUT")) else f"{DATA_OUT}/gen_{tag}_summary.json", "w"), indent=1)
