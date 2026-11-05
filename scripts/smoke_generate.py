"""Smoke test: base vs edited generation on a few GSM8K-style problems, thinking on, Astra's sampling."""
import sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
probs = [
 "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
 "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
 "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
]
def rep_ratio(ids, n=20):
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n)]
    return 1 - len(set(grams)) / max(1, len(grams))
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
for tag in sys.argv[1].split(","):
    path = f"/data2/ec2-user/models/qwen3.5-0.8b-astra/{tag}"
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    print(f"\n===== {tag}  ({type(m).__name__}) =====")
    for p in probs:
        msgs = [{"role": "user", "content": p + " Please reason step by step, and put your final answer within \\boxed{}."}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(txt, return_tensors="pt").to("cuda")
        torch.manual_seed(0); t = time.time()
        out = m.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
        gen = out[0, ids.input_ids.shape[1]:].tolist()
        s = tok.decode(gen, skip_special_tokens=False)
        print(f"--- {p[:50]}...  len={len(gen)} closed_think={'</think>' in s} boxed={'boxed' in s.split('</think>')[-1]} rep20={rep_ratio(gen):.2f}  {time.time()-t:.0f}s")
        print("   HEAD:", repr(s[:200])); print("   TAIL:", repr(s[-300:]))
    del m; torch.cuda.empty_cache()
