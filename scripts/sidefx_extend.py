"""Stage 2 of sidefx-confirm: continue cut samples of a 16k run to a total sequence length of 32,768 (Astra's max_model_len).

Same harness as sidefx_eval.py (same LLM config, eager, same hook install, same SAMPLING_THINK35, grade, loop_signals, classify).
Input records must carry prompt_ids + token_ids (sidefx_eval.py --save-ids). Each selected cut sample is continued from its exact
token ids; the combined text (16k text + continuation) is graded and classified like a fresh sample. Uncut records are copied.

    python scripts/sidefx_extend.py <model_dir> <in.jsonl> <tag> [--spec spec.json] [--k 0] [--total 32768] [--seed 2] --outdir D
"""
import os, sys, json, time, argparse
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sidefx_eval import install_hooks, classify


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("inp"); ap.add_argument("tag")
    ap.add_argument("--spec", default=None); ap.add_argument("--k", type=int, default=0); ap.add_argument("--total", type=int, default=32768)
    ap.add_argument("--mem", type=float, default=0.85); ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2); ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outp = f"{a.outdir}/{a.tag}.jsonl"
    assert not os.path.exists(outp), f"refusing to overwrite {outp}"
    from astra_eval import grade, grade_lenient, loop_signals, SAMPLING_THINK35
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    recs = [x for x in (json.loads(l) for l in open(a.inp)) if x["k"] == a.k]
    todo = [x for x in recs if x["cut"]]
    print(f"{len(recs)} records with k={a.k}, {len(todo)} cut -> extend to {a.total}", flush=True)
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem, max_model_len=a.total, enforce_eager=True, dtype="bfloat16",
              max_num_seqs=a.max_seqs, compilation_config={"mode": 0, "cudagraph_mode": "NONE"}, enable_prefix_caching=False)
    spec = json.load(open(a.spec)) if a.spec else None
    print("HOOKS", install_hooks(llm, spec) if spec else "no hooks", flush=True)
    sp_kw = dict(SAMPLING_THINK35); sp_kw["seed"] = a.seed
    prompts = [TokensPrompt(prompt_token_ids=x["prompt_ids"] + x["token_ids"]) for x in todo]
    sps = [SamplingParams(n=1, **{**sp_kw, "max_tokens": a.total - len(x["prompt_ids"]) - len(x["token_ids"])}) for x in todo]
    t0 = time.time(); outs = llm.generate(prompts, sps) if todo else []; dt = time.time() - t0
    new = {}
    ntok = 0
    for x, o in zip(todo, outs):
        s = o.outputs[0]; text = x["text"] + s.text; cut = s.finish_reason == "length"; ntok += len(s.token_ids)
        ok, pred = grade(text, x["gold"]); ls = loop_signals(text)
        y = {k: v for k, v in x.items() if k not in ("prompt_ids", "token_ids")}
        y.update(pred=pred, correct=ok, correct_lenient=grade_lenient(text, x["gold"]), n_tokens=x["n_tokens"] + len(s.token_ids),
                 cut=cut, closed="</think>" in text, loop_end=bool(ls["verbatim_loop_end"] or ls["repetitive_end"]),
                 cls=classify(text, ok, pred, cut, loop_signals), text=text, extended=True, cls_16k=x["cls"], correct_16k=x["correct"])
        new[(x["set"], x["id"])] = y
    with open(outp, "w") as f:
        for x in recs:
            y = new.get((x["set"], x["id"])) or {**{k: v for k, v in x.items() if k not in ("prompt_ids", "token_ids")},
                                                  "extended": False, "cls_16k": x["cls"], "correct_16k": x["correct"]}
            f.write(json.dumps(y) + "\n")
    R = [json.loads(l) for l in open(outp)]
    print("SUMMARY", json.dumps(dict(tag=a.tag, n=len(R), extended=len(todo), sec=round(dt), tok_per_s=round(ntok / max(dt, 1)),
          acc=sum(x["correct"] for x in R) / len(R), completion=sum(not x["cut"] for x in R) / len(R),
          loop_end=sum(x["loop_end"] for x in R) / len(R), ws=sum(x["cls"] == "cut_whitespace" for x in R),
          nobox_fin=sum((not x["cut"]) and x["pred"] is None for x in R))), flush=True)


if __name__ == "__main__":
    main()
