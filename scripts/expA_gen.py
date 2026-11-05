"""Experiment A: thinking-on generations with one harness for every model (vLLM, Astra THINK35 sampling, cap 12,288 tokens).
Problems: GSM8K = Astra's 200 logged ids + the next unused test ids in file order up to 500; MATH-500 = all 500.

    python scripts/expA_gen.py MODEL_DIR TAG [N_PER_SET]      -> $BLF_DATA/expA/gen_{TAG}.jsonl  (refuses to overwrite)
"""
import json, os, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import REPO, astra_rows, user_turn

MAXTOK = 12288


def problems(n_per_set=500):
    astra = {r["id"] for ds in ("gsm8k",) for r in astra_rows("unedited", ds)}
    g = [json.loads(l) for l in open(f"{REPO}/data/gsm8k_test.jsonl")]
    gold = lambda a: a.split("####")[-1].strip().replace(",", "")
    gs = [r for r in g if r["id"] in astra] + [r for r in g if r["id"] not in astra]
    out = [dict(id=r["id"], ds="gsm8k", problem=r["question"], answer=gold(r["answer"])) for r in gs[:n_per_set]]
    m = [json.loads(l) for l in open(f"{REPO}/data/math500.jsonl")]
    out += [dict(id=r["id"], ds="math500", problem=r["problem"], answer=r["answer"]) for r in m[:n_per_set]]
    return out


def main():
    model_dir, tag = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    outd = f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/expA"; os.makedirs(outd, exist_ok=True)
    of = f"{outd}/gen_{tag}.jsonl"
    assert not os.path.exists(of), f"refusing to overwrite {of}"
    rows = problems(n)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=model_dir, gpu_memory_utilization=0.88, max_model_len=MAXTOK + 1024, dtype="bfloat16", max_num_seqs=256, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_turn(r["problem"])}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True) for r in rows]
    print("PROMPT EXAMPLE:\n" + prompts[0], flush=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=MAXTOK, seed=0)
    outs = llm.generate(prompts, sp)
    with open(of, "w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            f.write(json.dumps(dict(r, text=c.text, n_tokens=len(c.token_ids), cut=c.finish_reason == "length", model=model_dir)) + "\n")
    sec = time.time() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"SUMMARY tag={tag} n={len(rows)} tokens={toks} sec={sec:.0f} cut={sum(o.outputs[0].finish_reason == 'length' for o in outs) / len(outs):.2f}", flush=True)


if __name__ == "__main__":
    main()
