"""Test 3c spot check: compare the Opus spot-check run (subset) with the main Sonnet run on the same neurons:
per-neuron AP differences (description and controls, both negative sets) and whether the two descriptions name the same trigger
(judged by the main judge model, Yes/No).
    python scripts/test3_explain_compare.py <main.json> <spot.json> [judge_model]   out: notes/test3_autointerp_compare_<RUNTAG>.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import REPO, RUNTAG

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); judge = sys.argv[3] if len(sys.argv) > 3 else "claude-sonnet-5-5"
ia = {(n["layer"], n["unit"]): n for n in A["neurons"]}; pairs = [(ia[(n["layer"], n["unit"])], n) for n in B["neurons"] if (n["layer"], n["unit"]) in ia]
from openai import OpenAI
c = OpenAI(base_url=os.environ["VECTOR_BASE_URL"], api_key=os.environ["VECTOR_API_KEY"], max_retries=6, timeout=120)
def same(a, b):
    p = (f"Two descriptions of the same neuron of a language model:\nA: {a}\nB: {b}\nIgnore wording, examples and minor details. "
         "Do both name the same main trigger (same token or pattern, in the same kind of text / situation)? Answer Yes or No.")
    r = c.chat.completions.create(model=judge, messages=[{"role": "user", "content": p}], max_tokens=1000)
    import re; v = re.findall(r"\b(yes|no)\b", (r.choices[0].message.content or "").lower()); return bool(v) and v[-1] == "yes"   # verdict = last Yes/No (the model explains first)
rows = []
for a, b in pairs:
    r = dict(layer=a["layer"], unit=a["unit"], is_random=a["is_random"], desc_main=a["desc"], desc_spot=b["desc"], same_trigger=same(a["desc"], b["desc"]))
    for k in ["ap_desc_negm", "ap_desc_negn", "ap_desc_loop_negn", "ap_desc_shuf_negn", "ap_desc_seg_negm"]:
        r[k + "_main"], r[k + "_spot"] = a.get(k), b.get(k)
    rows.append(r)
def m(k, sel): v = [x[k] for x in rows if sel(x) and x[k] is not None]; return float(np.mean(v)) if v else None
S = {}
for g, sel in [("astra", lambda x: not x["is_random"]), ("random", lambda x: x["is_random"])]:
    S[g] = dict(n=sum(sel(x) for x in rows), same_trigger_frac=m("same_trigger", sel),
                **{f"{k}_{w}": m(f"{k}_{w}", sel) for k in ["ap_desc_negm", "ap_desc_negn", "ap_desc_loop_negn", "ap_desc_shuf_negn"] for w in ["main", "spot"]})
print(json.dumps(S, indent=1))
json.dump(dict(main=sys.argv[1], spot=sys.argv[2], judge=judge, summary=S, neurons=rows),
          open(f"{REPO}/notes/test3_autointerp_compare_{RUNTAG}.json", "w"), indent=1)
