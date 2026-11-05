"""Experiment A: teacher-forced doubt signals at candidate-answer points.

    python scripts/expA_measure.py MODEL_DIR MODEL_TAG TRACES_TAG [MAX_TRACES]
TRACES_TAG = a generation file tag (gen_{TAG}.jsonl in $BLF_DATA/expA) or astra:{unedited|p50c}.
Out: $BLF_DATA/expA/meas_{MODEL_TAG}_on_{TRACES_TAG}.jsonl (refuses to overwrite). One row per candidate point.
Same detection / prefix / token sets / scoring for every model (scripts/expA_common.py).
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import REPO, DSETS, astra_rows, questions, user_turn, candidates, grade_value, measure_prefix, real_recheck, token_sets
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV = os.environ.get("EXPA_DEV", "cuda")
# Follow-up (thinking on vs off; not pre-registered): EXPA_NOTHINK=1 -> chat template with enable_thinking=False;
# EXPA_PAD_END=1 -> if a trace ends right after the candidate, measure the next token after the candidate + "\n\n"
# (the counterfactual "continue or stop" point) instead of dropping the candidate.
NOTHINK = os.environ.get("EXPA_NOTHINK") == "1"; PAD_END = os.environ.get("EXPA_PAD_END") == "1"
model_dir, mtag, ttag = sys.argv[1], sys.argv[2], sys.argv[3]
MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
D = f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/expA"; os.makedirs(D, exist_ok=True)
of = f"{D}/meas_{mtag}_on_{ttag.replace(':', '-')}.jsonl"
assert not os.path.exists(of), f"refusing to overwrite {of}"

if ttag == "t38nothink":   # base thinking-off generations from test 8 (Astra's 200 GSM8K + 200 MATH-500 ids)
    Q = questions()
    rows = [dict(id=r["id"], ds="gsm8k" if r["id"].startswith("gsm8k") else "math500", problem=Q[r["id"]], answer=r["answer"], text=r["text"])
            for r in map(json.loads, open(f"{os.environ.get('BLF_DATA', '/scratch/yang0531/blf_data')}/t38/gen_base_nothink.jsonl"))]
elif ttag.startswith("astra:"):
    Q = questions(); rows = []
    for ds in DSETS:
        for r in astra_rows(ttag.split(":")[1], ds): rows.append(dict(id=r["id"], ds=ds, problem=Q[r["id"]], answer=r["answer"], text=r["text"]))
else:
    rows = [json.loads(l) for l in open(f"{D}/gen_{ttag}.jsonl")]
rows = rows[:MAXN]

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(DEV).eval()
dmask, wmask, amask, demask, wemask = (torch.tensor(m, device=DEV) for m in token_sets(tok))
small = model.config.get_text_config().hidden_size == 1024
U = {L: torch.tensor(np.load(f"{REPO}/shared/u_L{L}.npy"), dtype=torch.float32, device=DEV) for L in (15, 19)} if small else None
print("model", model_dir, "hidden", model.config.get_text_config().hidden_size, "u-readout", small,
      "| set sizes doubt/wrap/ambig", int(dmask.sum()), int(wmask.sum()), int(amask.sum()), flush=True)

t0 = time.time(); nout = 0
with open(of, "w") as f, torch.no_grad():
    for ri, r in enumerate(rows):
        think = r["text"].split("</think>")[0]
        pts = []
        for ci, c in enumerate(candidates(think, 3)):
            pre = measure_prefix(think, c)
            if pre is None and PAD_END:            # trace ends on the candidate's line: keep that line, add a blank line
                think = think.rstrip() + "\n\n"; pts.append((ci, c, len(think))); break
            if pre is None: continue
            pts.append((ci, c, len(pre)))
        if not pts: continue
        prompt = tok.apply_chat_template([{"role": "user", "content": user_turn(r["problem"])}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=not NOTHINK)
        pid = tok(prompt).input_ids
        enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
        gid, offs = enc.input_ids, enc.offset_mapping
        # token index j of the generated text whose span holds the measurement char k (first content char of the next line)
        js = []
        for ci, c, k in pts:
            j = next((t for t, (a, b) in enumerate(offs) if b > k), None)
            if j is None and PAD_END and k == len(think): j = len(gid)      # measure after the whole (padded) text
            if j is None or j == 0: continue
            js.append((ci, c, k, j))
        if not js: continue
        jmax = max(j for *_, j in js)
        ids = torch.tensor([pid + gid[:jmax]], device=DEV)
        out = model.model(ids, output_hidden_states=small)
        P = len(pid)
        LOG = model.lm_head(out.last_hidden_state[0, [P + j - 1 for *_, j in js]]).float()
        if nout == 0:   # sanity: lm_head(last_hidden_state) == full forward logits at the last position
            ref = model(ids, logits_to_keep=1).logits[0, -1].float()
            print("sanity max|diff| last-pos logits:", float((model.lm_head(out.last_hidden_state[0, -1]).float() - ref).abs().max()), flush=True)
        for qi, (ci, c, k, j) in enumerate(js):
            pos = P + j - 1                       # logits at pos predict generated token j
            lp = torch.log_softmax(LOG[qi], -1)[: len(dmask)]
            p = lp.exp()
            pd, pw, pa = float(p[dmask].sum()), float(p[wmask].sum()), float(p[amask].sum())
            top = torch.topk(p, 8)
            rec = dict(id=r["id"], ds=r["ds"], model=mtag, traces=ttag, cand_idx=ci, kind=c["kind"], value=c["value"], gold=r["answer"],
                       correct=grade_value(c["value"], r["answer"]), char_pos=c["start"], tok_pos=j, real_recheck=real_recheck(think, c),
                       next_tok_real=tok.decode([gid[j]]) if j < len(gid) else "<END>", p_real_next=float(p[gid[j]]) if j < len(gid) and gid[j] < len(p) else None,
                       nothink_template=NOTHINK, padded_end=j >= len(gid),
                       p_doubt=pd, p_wrap=pw, p_let=pa, ratio=pd / max(pd + pw, 1e-12),
                       p_doubt_ext=float(p[demask].sum()), p_wrap_ext=float(p[wemask].sum()),
                       top=[(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
                       ctx=think[max(0, c["start"] - 200):k] + "【】" + think[k:k + 120])
            if small:
                hs = out.hidden_states
                rec.update(u15_hs16=float(hs[16][0, pos].float() @ U[15]), u15_hs20=float(hs[20][0, pos].float() @ U[15]),
                           u19_hs20=float(hs[20][0, pos].float() @ U[19]))
            f.write(json.dumps(rec) + "\n"); nout += 1
        del out
        if ri % 50 == 0: print(f"{ri}/{len(rows)} traces, {nout} points, {time.time() - t0:.0f}s", flush=True)
print(f"SUMMARY {of} points={nout} sec={time.time() - t0:.0f}", flush=True)
