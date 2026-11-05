"""Do hooks survive CUDA-graph capture? Greedy decode of 16 GSM8K prompts x 512 tokens (+ a 256-prompt x 1024-token timing batch)
with a spec, under cudagraph mode none (eager) or full_decode_only.  python scripts/sidefx_cgtest.py <model> <spec|none> <mode> <out.json>"""
import os, sys, json, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sidefx_eval import install_hooks, REPO
from vllm import LLM, SamplingParams
mode = sys.argv[3]
llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.85, max_model_len=32768, enforce_eager=(mode == "none"), dtype="bfloat16",
          max_num_seqs=256, compilation_config={"mode": 0, "cudagraph_mode": mode.upper()}, enable_prefix_caching=False)
if sys.argv[2] != "none": print("HOOKS", install_hooks(llm, json.load(open(sys.argv[2]))), flush=True)
tok = llm.get_tokenizer()
rows = [json.loads(l) for l in open(f"{REPO}/data/gsm8k_test.jsonl")]
P = [tok.apply_chat_template([{"role": "user", "content": r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows[:256]]
o = llm.generate(P[:16], SamplingParams(max_tokens=512, temperature=0))
ids = [list(x.outputs[0].token_ids) for x in o]
t0 = time.time(); o2 = llm.generate(P, SamplingParams(max_tokens=1024, temperature=0.6, top_p=0.95, top_k=20, seed=0)); dt = time.time() - t0
ntok = sum(len(x.outputs[0].token_ids) for x in o2)
json.dump(dict(ids=ids, tok_per_s=ntok / dt, mode=mode), open(sys.argv[4], "w"))
print(f"CGTEST mode={mode} tok/s={ntok/dt:.0f}  sample: {tok.decode(ids[0][-120:])!r}", flush=True)
