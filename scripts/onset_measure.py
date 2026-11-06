"""Loop-onset analysis, stage 2 (GPU): resid·u at every stated answer (Exp A detector) before onset (looping traces) and in
non-looping traces. Measurement identical to scripts/expA_measure.py (measure_prefix; token covering the first content char of the
next line; hidden_states[16]·u15, hidden_states[20]·u15/u19).

    python scripts/onset_measure.py SOURCE EXTRACT_JSONL OUT [MAX_TRACES]
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import REPO, user_turn, measure_prefix
from onset_extract import load
from transformers import AutoTokenizer, AutoModelForCausalLM

source, ef, of = sys.argv[1], sys.argv[2], sys.argv[3]
MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
assert not os.path.exists(of), f"refusing to overwrite {of}"
MODEL = "/scratch/yang0531/blf_models/base"
E = {r["id"]: r for r in map(json.loads, open(ef))}
rows = [r for r in load(source) if r["id"] in E][:MAXN]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
U = {L: torch.tensor(np.load(f"{REPO}/shared/u_L{L}.npy"), dtype=torch.float32, device="cuda") for L in (15, 19)}
t0 = time.time(); nout = 0
with open(of, "w") as f, torch.no_grad():
    for ri, r in enumerate(rows):
        e = E[r["id"]]
        group = "loop_pre_onset" if e["loops"] else ("finish" if e["closed"] else "cut_noloop")
        text = r["text"]
        pts = []
        for ai, a in enumerate(e["answers"]):
            pre = measure_prefix(text, a)
            if pre is None: continue
            pts.append((ai, a, len(pre)))
        if not pts: continue
        prompt = tok.apply_chat_template([{"role": "user", "content": user_turn(r["problem"])}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        pid = tok(prompt).input_ids
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        gid, offs = enc.input_ids, enc.offset_mapping
        ends = [b for a, b in offs]
        import bisect
        js = []
        for ai, a, k in pts:
            j = bisect.bisect_right(ends, k)          # first token whose span end > k  (== expA_measure's rule)
            if j >= len(gid) or j == 0: continue
            js.append((ai, a, j))
        if not js: continue
        jmax = max(j for *_, j in js)
        ids = torch.tensor([pid + gid[:jmax]], device="cuda")
        out = model.model(ids, output_hidden_states=True)
        hs = out.hidden_states; P = len(pid)
        for ai, a, j in js:
            pos = P + j - 1
            f.write(json.dumps(dict(id=r["id"], ds=r["ds"], source=source, group=group, ans_idx=ai, n_ans=len(e["answers"]),
                                    last_before_onset=e["loops"] and ai == len(e["answers"]) - 1, value=a["value"], correct=a["correct"],
                                    tok_pos=j, onset_tok=e["onset_tok"], cls=e.get("cls"), next_tok_real=tok.decode([gid[j]]),
                                    u15_hs16=float(hs[16][0, pos].float() @ U[15]), u15_hs20=float(hs[20][0, pos].float() @ U[15]),
                                    u19_hs20=float(hs[20][0, pos].float() @ U[19]))) + "\n"); nout += 1
        del out, hs
        if ri % 100 == 0: print(f"{ri}/{len(rows)} traces, {nout} points, {time.time() - t0:.0f}s", flush=True)
print(f"SUMMARY {of} points={nout} sec={time.time() - t0:.0f}", flush=True)
