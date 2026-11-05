"""Screening-set generation eval with Astra's protocol (prompt, sampling think35, grader, loop rule), vLLM, low GPU share.
    python scripts/screen_eval.py <model_dir> <tag> [--sets gsm8k,math500] [--n 100] [--samples 1] [--max-tokens 16384]
        [--no-think] [--rep-penalty 1.1] [--think-budget 4096] [--mem 0.07]
Writes notes/screen/<tag>.jsonl (+ summary line in notes/screen/summary.jsonl)."""
import os, sys, json, time, argparse
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")

def load_sets(names, n):
    out = {}
    for name in names:
        if name == "gsm8k":
            rows = [json.loads(l) for l in open("data/gsm8k_test.jsonl")]; rows = [dict(id=r["id"], problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
        elif name == "math500":
            rows = [json.loads(l) for l in open("data/math500.jsonl")]
        else: raise ValueError(name)
        out[name] = rows[:n]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--sets", default="gsm8k,math500"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--no-think", action="store_true"); ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--think-budget", type=int, default=0); ap.add_argument("--mem", type=float, default=float(os.environ.get("EVAL_GPU_MEM", 0.85))); ap.add_argument("--max-seqs", type=int, default=int(os.environ.get("EVAL_MAX_SEQS", 256))); ap.add_argument("--eager", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from astra_eval import build_prompt, grade, grade_lenient, loop_signals, SAMPLING_THINK35
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem, max_model_len=32768, enforce_eager=a.eager, dtype="bfloat16", max_num_seqs=a.max_seqs)
    tok = llm.get_tokenizer()
    sets = load_sets(a.sets.split(","), a.n)
    sp_kw = dict(SAMPLING_THINK35); sp_kw["max_tokens"] = a.max_tokens; sp_kw["seed"] = a.seed
    if a.rep_penalty != 1.0: sp_kw["repetition_penalty"] = a.rep_penalty
    os.makedirs("notes/screen", exist_ok=True); allrecs = []; t0 = time.time(); ntok = 0
    for name, rows in sets.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r["problem"])}], tokenize=False, add_generation_prompt=True, enable_thinking=not a.no_think) for r in rows]
        if a.think_budget:
            # stage 1: think up to the budget; stage 2: force "</think>\n\n" and answer with the remaining tokens
            s1 = SamplingParams(n=a.samples, **{**sp_kw, "max_tokens": a.think_budget})
            o1 = llm.generate(prompts, s1)
            p2, meta = [], []
            for p, o in zip(prompts, o1):
                for s in o.outputs:
                    if "</think>" in s.text: p2.append(None); meta.append((s.text, len(s.token_ids), False))
                    else: p2.append(p + s.text + "\n</think>\n\n"); meta.append((s.text + "\n</think>\n\n", len(s.token_ids), True))
            s2 = SamplingParams(n=1, **{**sp_kw, "max_tokens": a.max_tokens - a.think_budget})
            o2 = llm.generate([p for p in p2 if p], s2); it = iter(o2)
            texts = []
            for p, (t, n, forced) in zip(p2, meta):
                if p: s = next(it).outputs[0]; texts.append((t + s.text, n + len(s.token_ids), s.finish_reason == "length"))
                else: texts.append((t, n, False))
            groups = [texts[i*a.samples:(i+1)*a.samples] for i in range(len(rows))]
        else:
            outs = llm.generate(prompts, SamplingParams(n=a.samples, **sp_kw))
            groups = [[(s.text, len(s.token_ids), s.finish_reason == "length") for s in o.outputs] for o in outs]
        for r, g in zip(rows, groups):
            for k, (text, n, cut) in enumerate(g):
                ok, pred = grade(text, r["answer"]); ls = loop_signals(text); ntok += n; okl = grade_lenient(text, r["answer"])
                allrecs.append(dict(set=name, id=r["id"], k=k, gold=r["answer"], pred=pred, correct=ok, correct_lenient=okl, n_tokens=n, cut=cut,
                                    closed=("</think>" in text) or a.no_think, loop_end=bool(ls["verbatim_loop_end"] or ls["repetitive_end"]), text=text))
    dt = time.time() - t0
    with open(f"notes/screen/{a.tag}.jsonl", "w") as f:
        for x in allrecs: f.write(json.dumps(x) + "\n")
    summ = dict(tag=a.tag, args=vars(a), sec=round(dt), tok_per_s=round(ntok / dt))
    for name in sets:
        R = [x for x in allrecs if x["set"] == name]; done = [x for x in R if not x["cut"]]
        summ[name] = dict(n=len(R), completion=len(done)/len(R), acc_all=sum(x["correct"] for x in R)/len(R), acc_on_completed=sum(x["correct"] for x in done)/max(1,len(done)), acc_lenient=sum(x["correct_lenient"] for x in R)/len(R), no_box_finished=sum(1 for x in done if x["pred"] is None)/len(R),
                          loop_end=sum(x["loop_end"] for x in R)/len(R), mean_tokens=sum(x["n_tokens"] for x in R)/len(R))
    print("SUMMARY", json.dumps(summ))
    with open("notes/screen/summary.jsonl", "a") as f: f.write(json.dumps(summ) + "\n")

if __name__ == "__main__":
    main()
