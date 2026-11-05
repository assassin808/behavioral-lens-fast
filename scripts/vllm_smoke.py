"""Can vLLM run the 0.8B model next to the 30B server?  Offline generation, low memory fraction."""
import sys, time, os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "p50c"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.07
    llm = LLM(model=f"/data2/ec2-user/models/qwen3.5-0.8b-astra/{tag}", gpu_memory_utilization=frac, max_model_len=32768,
              enforce_eager=True, dtype="bfloat16", max_num_seqs=16)
    tok = llm.get_tokenizer()
    qs = ["Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
          "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + " Please reason step by step, and put your final answer within \\boxed{}."}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for q in qs]
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768, seed=0)
    t = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t
    for o in outs:
        s = o.outputs[0].text; n = len(o.outputs[0].token_ids)
        print(f"len={n} closed={'</think>' in s} boxed={'boxed' in s.split('</think>')[-1]} finish={o.outputs[0].finish_reason}")
        print("  TAIL:", repr(s[-300:]))
    print(f"total {sum(len(o.outputs[0].token_ids) for o in outs)} tokens in {dt:.0f}s")

if __name__ == "__main__":
    main()
