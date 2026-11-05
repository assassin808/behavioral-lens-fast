"""Base model, thinking OFF, on the same GSM8K / MATH-500 problems Astra logged first samples for (200 + 200), Astra's sampling
(think35: T 0.6, top-p 0.95, top-k 20, presence 1.5), 16k cap, 1 sample. Used by test 8 (does u light up when no-think loops?) and the
thinking-on vs -off comparison along u.  Out: $T38_DATA/gen_base_nothink.jsonl (gitignored) + one summary line printed.

    python scripts/test8_gen_nothink.py [model_dir] [tag]
"""
import os, sys, json, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODEL_DIR, RUNTAG, weights_file, DATA_OUT, astra_rows, question, loop_mask

model_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR["base"]
tag = sys.argv[2] if len(sys.argv) > 2 else ("base_nothink" + ("_pilot" if os.environ.get("PILOT") else ""))
BACKEND = os.environ.get("GEN_BACKEND", "vllm")      # 'hf' = transformers fallback where vLLM is unavailable (no presence penalty, 8k cap)
rows = astra_rows("unedited", "gsm8k") + astra_rows("unedited", "math500")
if os.environ.get("PILOT"): rows = rows[:8] + rows[200:208]
t0 = time.time()
if BACKEND == "vllm":
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, gpu_memory_utilization=0.85, max_model_len=20480, dtype="bfloat16", max_num_seqs=256, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": question(r["id"])}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for r in rows]
    outs = [(list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].finish_reason == "length") for o in
            llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=16384, seed=0))]
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": question(r["id"])}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for r in rows]
    MAXNEW, BS = int(os.environ.get("GEN_MAXNEW", 8192)), int(os.environ.get("GEN_BS", 100))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])); res = {}
    torch.manual_seed(0)
    for b0 in range(0, len(order), BS):
        idx = order[b0:b0 + BS]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idx):
            t = g[k, enc.input_ids.shape[1]:].tolist(); eos = [j for j, x in enumerate(t) if x in (tok.eos_token_id, tok.pad_token_id) or x == tok.convert_tokens_to_ids("<|im_end|>")]
            cut = not eos; t = t[:eos[0]] if eos else t
            res[i] = (t, tok.decode(t, skip_special_tokens=True), cut)
        print(f"hf gen batch {b0 // BS + 1}: {time.time() - t0:.0f}s", flush=True)
    outs = [res[i] for i in range(len(prompts))]
try:
    from astra_eval import grade
except Exception:
    grade = None
os.makedirs(DATA_OUT, exist_ok=True)
recs = []
for r, (ids, text, cut) in zip(rows, outs):
    m = loop_mask(ids)
    try: ok = grade(text, r["answer"])[0] if (grade and r["answer"] is not None) else None
    except Exception: ok = None
    recs.append(dict(id=r["id"], src=r["src"], answer=r["answer"], text=text, n_tokens=len(ids), cut=cut, backend=BACKEND,
                     loop_frac=float(m.mean()) if len(m) else 0.0, correct=ok))
with open(f"{DATA_OUT}/gen_{tag}.jsonl", "w") as f:
    for x in recs: f.write(json.dumps(x) + "\n")
summ = dict(tag=tag, backend=BACKEND, sec=round(time.time() - t0))
for src in ["gsm8k", "math"]:
    R = [x for x in recs if x["src"] == src]
    summ[src] = dict(n=len(R), cut=sum(x["cut"] for x in R) / len(R), loop_frac_gt_5pct=sum(x["loop_frac"] > .05 for x in R) / len(R),
                     mean_tokens=sum(x["n_tokens"] for x in R) / len(R),
                     acc=(sum(bool(x["correct"]) for x in R) / len(R)) if grade else None) if R else {}
print("SUMMARY", json.dumps(summ), flush=True)
json.dump(summ, open(f"{DATA_OUT}/gen_{tag}_summary.json", "w"), indent=1)
