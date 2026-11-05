"""Equivalence check: teacher-forced token logprobs of <model> (+ optional hook spec) on 8 of Astra's logged p50c GSM8K texts
(first 3k tokens), through the same vLLM path as sidefx_eval.py.  python scripts/sidefx_check.py <model> <spec|none> <out.npy>"""
import os, sys, json, gzip, glob
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sidefx_eval import install_hooks, REPO
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.5, max_model_len=8192, enforce_eager=True, dtype="bfloat16",
          compilation_config={"mode": 0, "cudagraph_mode": "NONE"}, enable_prefix_caching=False)
if sys.argv[2] != "none": print("HOOKS", install_hooks(llm, json.load(open(sys.argv[2]))))
tok = llm.get_tokenizer()
rows = []
for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/p50c/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz")):
    rows += [json.loads(l) for l in gzip.open(f, "rt")]
rows = rows[:8]
P = [TokensPrompt(prompt_token_ids=tok(r["text"], add_special_tokens=False).input_ids[:3000]) for r in rows]
outs = llm.generate(P, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0))
lp = [np.array([list(d.values())[0].logprob if d else 0.0 for d in o.prompt_logprobs[1:]]) for o in outs]
np.save(sys.argv[3], np.concatenate(lp)); print("saved", sys.argv[3], sum(len(x) for x in lp))
