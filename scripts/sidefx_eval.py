"""Side-effect dissection of Astra's p50c edit: one shared vLLM harness for ALL arms (base, p50c, and hook-based variants).

Same prompt / sampling / grader / loop rule as scripts/screen_eval.py (Astra's protocol, thinking on). Differences, applied to
every arm alike: vLLM runs without torch.compile (and optionally without CUDA graphs) so that PyTorch hooks execute, and each arm
may carry an edit spec that is applied as a hook on the BASE weights.

Edit spec (JSON): {"layers": {"15": {"cols": [...], "alpha": float | [per-col floats], "floor": null | float}, "19": {...}}}
  At decoder layer L the hook adds  d(t) * u_L  to the layer output, where
      d(t) = - sum_j alpha_j * act_j(t) * (u_L^T w_j)          (exactly the rank-1 edit w_j' = w_j - alpha_j u u^T w_j)
  (implemented with the bf16-rounded edited columns, dW = bf16(w - alpha u u^T w) - w, so alpha = 6 matches the real p50c weights)
  With a floor F, the residual-stream projection after layer L is not pushed below F by the edit:
      p = (resid after L) . u_L  (base weights);   p_new = max(p + d, min(p, F));   add (p_new - p) * u_L
  (positions whose base projection is already below F are left as they are; the edit never pushes anything up).

    python scripts/sidefx_eval.py <model_dir> <tag> [--spec spec.json] [--sets gsm8k,math500] [--n 20] [--offset 0]
        [--samples 1] [--max-tokens 16384] [--cudagraph none|full_decode_only] [--outdir notes/sidefx]
Writes <outdir>/<tag>.jsonl (refuses to overwrite) and appends a summary line to <outdir>/summary.jsonl.
"""
import os, sys, json, time, argparse, re
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")      # model lives in this process -> hooks can be registered
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sets(names, n, offset):
    out = {}
    for name in names:
        if name == "gsm8k":
            rows = [json.loads(l) for l in open(f"{REPO}/data/gsm8k_test.jsonl")]
            rows = [dict(id=r["id"], problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
        elif name == "math500":
            rows = [json.loads(l) for l in open(f"{REPO}/data/math500.jsonl")]
        else:
            raise ValueError(name)
        out[name] = rows[offset:offset + n]
    return out


def classify(text, ok, pred, cut, loop_signals):
    """Shared rule-based failure class (as scripts/failure_classes_from_logs.py on branch yang/results)."""
    if ok: return "correct"
    if cut:
        ls = loop_signals(text)
        if re.fullmatch(r"\s*", text[-2000:]): return "cut_whitespace"
        if ls["verbatim_loop_end"]: return "cut_verbatim_loop"
        if ls["repetitive_end"]: return "cut_repetitive"
        return "cut_still_reasoning"
    if "</think>" not in text: return "finished_no_think_close"
    if pred is None: return "finished_no_box"
    return "finished_wrong"


def install_hooks(llm, spec):
    """Register the edit hooks inside the (in-process) vLLM model. Returns a description string."""
    import numpy as np, torch
    L2 = {int(k): v for k, v in spec["layers"].items()}
    U = {L: np.load(f"{REPO}/shared/u_L{L}.npy").astype("float32") for L in L2}

    def fn(model):
        import torch
        mods = dict(model.named_modules())
        msgs = []
        for L, cfg in L2.items():
            lay = [n for n in mods if re.search(rf"(^|\.)layers\.{L}$", n)]
            assert len(lay) == 1, lay
            layer = mods[lay[0]]; down = layer.mlp.down_proj
            W = down.weight                                              # [1024, 3584] (out, in), TP=1
            dev = W.device
            u = torch.tensor(U[L], device=dev, dtype=torch.float32)
            cols = torch.tensor(cfg["cols"], device=dev, dtype=torch.long)
            a = cfg["alpha"]
            alpha = torch.tensor(a if isinstance(a, list) else [a] * len(cfg["cols"]), device=dev, dtype=torch.float32)
            Wc = W[:, cols].double()                                     # base columns (bf16 values)
            ud = u.double()
            Wn = (Wc - ud[:, None] * (alpha.double()[None, :] * (ud @ Wc)[None, :])).float().to(torch.bfloat16)
            dW = (Wn.float() - Wc.float())                               # exact bf16 weight change, as a real weight edit would make
            coef = u @ dW                                                # its u-projection per column, [ncols]
            floor = cfg.get("floor")
            store = {}

            def pre(mod, inp, store=store):
                store["act"] = inp[0]

            def post(mod, inp, out, u=u, cols=cols, dW=dW, floor=floor, store=store):
                hs, res = out
                act = store.pop("act").index_select(-1, cols).float()   # [T, ncols]
                delta = act @ dW.T                                       # [T, 1024]
                if floor is not None:
                    d = delta @ u
                    p = (hs.float() + res.float()) @ u
                    pn = torch.maximum(p + d, torch.minimum(p, torch.full_like(p, floor)))
                    delta = delta + (pn - p - d)[:, None] * u[None, :]
                return ((hs.float() + delta).to(hs.dtype), res)

            down.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)
            msgs.append(f"L{L} {lay[0]} cols={len(cfg['cols'])} alpha_mean={float(alpha.mean()):.3f} floor={floor} "
                        f"sum|coef|={float(coef.abs().sum()):.4f}")
        return "; ".join(msgs)

    return llm.apply_model(fn)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--spec", default=None); ap.add_argument("--sets", default="gsm8k,math500")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--max-seqs", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cudagraph", default="none"); ap.add_argument("--outdir", default=f"{REPO}/notes/sidefx")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    outp = f"{a.outdir}/{a.tag}.jsonl"
    assert not os.path.exists(outp), f"refusing to overwrite {outp}"
    from astra_eval import build_prompt, grade, grade_lenient, loop_signals, SAMPLING_THINK35
    from vllm import LLM, SamplingParams
    cc = {"mode": 0, "cudagraph_mode": a.cudagraph.upper()}
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem, max_model_len=32768, enforce_eager=(a.cudagraph == "none"),
              dtype="bfloat16", max_num_seqs=a.max_seqs, compilation_config=cc, enable_prefix_caching=False)
    spec = json.load(open(a.spec)) if a.spec else None
    hook_msg = install_hooks(llm, spec) if spec else "no hooks"
    print("HOOKS", hook_msg, flush=True)
    tok = llm.get_tokenizer()
    sets = load_sets(a.sets.split(","), a.n, a.offset)
    sp_kw = dict(SAMPLING_THINK35); sp_kw["max_tokens"] = a.max_tokens; sp_kw["seed"] = a.seed
    allrecs = []; t0 = time.time(); ntok = 0
    for name, rows in sets.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r["problem"])}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for r in rows]
        outs = llm.generate(prompts, SamplingParams(n=a.samples, **sp_kw))
        for r, o in zip(rows, outs):
            for k, s in enumerate(o.outputs):
                text, n, cut = s.text, len(s.token_ids), s.finish_reason == "length"
                ok, pred = grade(text, r["answer"]); ls = loop_signals(text); ntok += n
                allrecs.append(dict(set=name, id=r["id"], k=k, gold=r["answer"], pred=pred, correct=ok,
                                    correct_lenient=grade_lenient(text, r["answer"]), n_tokens=n, cut=cut, closed="</think>" in text,
                                    loop_end=bool(ls["verbatim_loop_end"] or ls["repetitive_end"]),
                                    cls=classify(text, ok, pred, cut, loop_signals), text=text))
    dt = time.time() - t0
    with open(outp, "w") as f:
        for x in allrecs: f.write(json.dumps(x) + "\n")
    summ = dict(tag=a.tag, args=vars(a), hooks=hook_msg, sec=round(dt), tok_per_s=round(ntok / dt))
    for name in sets:
        R = [x for x in allrecs if x["set"] == name]; done = [x for x in R if not x["cut"]]
        cls = {}
        for x in R: cls[x["cls"]] = cls.get(x["cls"], 0) + 1
        summ[name] = dict(n=len(R), completion=len(done) / len(R), acc_all=sum(x["correct"] for x in R) / len(R),
                          acc_on_completed=sum(x["correct"] for x in done) / max(1, len(done)),
                          acc_lenient=sum(x["correct_lenient"] for x in R) / len(R),
                          no_box_finished=sum(1 for x in done if x["pred"] is None) / len(R),
                          loop_end=sum(x["loop_end"] for x in R) / len(R), mean_tokens=sum(x["n_tokens"] for x in R) / len(R), classes=cls)
    print("SUMMARY", json.dumps(summ), flush=True)
    with open(f"{a.outdir}/summary.jsonl", "a") as f: f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
