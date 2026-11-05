"""Cross-family re-judging of the test-3c auto-interp (see notes/api_judge_crossfamily.md).
    python scripts/api/t3_crossjudge.py <explainer> <judge> [kinds] [limit]
explainer: 'sonnet' = reuse the stored claude-sonnet-5-5 descriptions; otherwise an API model name that writes new descriptions
           (cached) with the test-3 explain prompt.
judge    : API model name. kinds: comma list from desc,desc_shuf,desc_seg,desc_loop (default all). limit: first N neurons (pilot).
Windows, window order, shuffled-description permutation and prompts are identical to scripts/test3_explain.py.
out: notes/api_t3_{explainer}__{judge}[_lim{N}].json  (never overwritten: refuses if exists)
"""
import os, sys, json, random, re, concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apicache
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPL, JUDGE = sys.argv[1], sys.argv[2]
KINDS = (sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "all" else "desc,desc_shuf,desc_seg,desc_loop").split(",")
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
PR = json.load(open("/scratch/yang0531/blf_data/t38/t3_explain_prep_32768.json")); N = PR["neurons"]
SON = json.load(open(f"{REPO}/notes/test3_autointerp_32768_claude-sonnet-5-5_killarney.json"))
SD = {(x["layer"], x["unit"]): x for x in SON["neurons"]}
# ---- verbatim from scripts/test3_explain.py ----
SRCNAME = {"gsm8k": "GSM8K math reasoning", "math": "competition-math reasoning", "ifeval": "instruction-following task",
           "mmlupro": "multiple-choice question (MMLU-Pro)"}
CTX = ("We study a small language model (Qwen3.5-0.8B) writing long step-by-step reasoning for math problems, instruction-following "
       "tasks and multiple-choice questions. It often gets stuck repeating an earlier passage verbatim (a generation loop); each "
       "excerpt says whether the marked token lies inside such a verbatim-repeated passage.")
def explain_prompt(n):
    ex = "\n".join(f"{k + 1}. ({w['seg']}) ...{w['text']}..." for k, w in enumerate(n["ex"]))
    return (f"{CTX}\nBelow are text excerpts where one neuron of the model fires most strongly. Tokens where it fires are marked [[like this]].\n"
            f"{ex}\n\nWhen this neuron fires, it pushes the model to output tokens like: {', '.join(repr(t) for t in n['promotes'][:10])}\n\n"
            "In ONE short sentence (max 20 words), describe what this neuron detects or does. Be specific about the textual pattern "
            "(e.g. which token, in which kind of text, in what situation). Answer with just the sentence.")
def detect_batch(desc, ws):
    ex = "\n".join(f"{k + 1}. ({w['seg']}) ...{w['text']}..." for k, w in enumerate(ws))
    return (f"{CTX}\nA neuron of this model is described as: \"{desc}\"\n\nBelow are {len(ws)} text excerpts, each with ONE token marked "
            f"[[like this]]. Some are tokens where the neuron fires strongly, others are random tokens.\n{ex}\n\n"
            "For each excerpt, rate from 0 to 100 how likely it is that the neuron fires strongly on the marked token, judging only from "
            f"the description. Answer with exactly {len(ws)} lines of the form `<index>: <score>` and nothing else.")
# ------------------------------------------------
ok = [n for n in N if len(n["ex"]) >= 4 and len(n["pos"]) >= 4 and len(n["negm"]) >= 4]
assert [(n["layer"], n["unit"]) for n in ok] == [(x["layer"], x["unit"]) for x in SON["neurons"]], "neuron list differs from Sonnet run"
TH = int(os.environ.get("API_THREADS", 12))
def pmap(f, xs):
    with cf.ThreadPoolExecutor(TH) as ex: return list(ex.map(f, xs))
if EXPL == "sonnet":
    for n in ok: n["desc"] = SD[(n["layer"], n["unit"])]["desc"]
else:
    outs = pmap(lambda n: apicache.chat(EXPL, explain_prompt(n), 4000), ok)
    for n, o in zip(ok, outs): n["desc"] = (o or "").strip().split("\n")[0].strip()
perm = list(range(len(ok))); random.Random(0).shuffle(perm)
for i, n in enumerate(ok):
    n["desc_shuf"] = ok[perm[i] if perm[i] != i else (i + 1) % len(ok)]["desc"]
    s, lp = n["seg_main"]
    n["desc_seg"] = (f"Fires on tokens inside a passage that the model is repeating verbatim (a generation loop), in {SRCNAME[s]} text." if lp
                     else f"Fires on tokens in {SRCNAME[s]} text.")
    n["desc_loop"] = "Fires on tokens inside a passage that the model is repeating verbatim (a generation loop)."
    if EXPL == "sonnet":
        for kd in ["desc_shuf", "desc_seg", "desc_loop"]: assert n[kd] == SD[(n["layer"], n["unit"])][kd], (kd, i)
sel = ok if LIMIT is None else ok[:LIMIT // 2] + [x for x in ok if x["is_random"]][:LIMIT - LIMIT // 2]
items, keys = [], []
for i, n in enumerate(ok):
    if not any(n is s for s in sel): continue
    ws = [(1, "pos", w) for w in n["pos"]] + [(0, "negm", w) for w in n["negm"]] + [(0, "negn", w) for w in n["negn"]]
    random.Random(1000 + i).shuffle(ws)
    for kd in KINDS:
        items.append((n[kd], [w for _, _, w in ws])); keys.append((i, kd, [(l, g) for l, g, _ in ws]))
raw = pmap(lambda it: apicache.chat(JUDGE, detect_batch(*it), 4000), items)
missing = 0; scores = []
for (d, ws), o in zip(items, raw):
    sc = {int(a): float(b) for a, b in re.findall(r"^\s*`?(\d+)\s*[:.)-]\s*(\d+(?:\.\d+)?)", o or "", re.M)}
    scores.append([min(max(sc.get(k + 1, 50.0), 0), 100) / 100 for k in range(len(ws))]); missing += sum(k + 1 not in sc for k in range(len(ws)))
from sklearn.metrics import average_precision_score
def score(pos, neg):
    if len(neg) < 2 or len(pos) < 2: return None, None
    return float((np.mean([p > .5 for p in pos]) + np.mean([p <= .5 for p in neg])) / 2), float(average_precision_score([1] * len(pos) + [0] * len(neg), pos + neg))
out = {}
for (i, kd, lab), sc in zip(keys, scores):
    n = ok[i]; r = out.setdefault(i, dict(layer=n["layer"], unit=n["unit"], is_random=n["is_random"], desc=n["desc"], desc_sonnet=SD[(n["layer"], n["unit"])]["desc"]))
    pos = [s for (l, g), s in zip(lab, sc) if g == "pos"]
    for g in ["negm", "negn"]:
        r[f"bacc_{kd}_{g}"], r[f"ap_{kd}_{g}"] = score(pos, [s for (l, gg), s in zip(lab, sc) if gg == g])
out = list(out.values())
def summ(xs):
    s = dict(n=len(xs))
    for kd in KINDS:
        for g in ["negm", "negn"]:
            v = [x[f"ap_{kd}_{g}"] for x in xs if x.get(f"ap_{kd}_{g}") is not None]; s[f"ap_{kd}_{g}"] = float(np.mean(v)) if v else None
    return s
S = {g: summ([x for x in out if f(x)]) for g, f in [("astra_L15", lambda x: not x["is_random"] and x["layer"] == 15),
                                                     ("astra_L19", lambda x: not x["is_random"] and x["layer"] == 19),
                                                     ("random", lambda x: x["is_random"])]}
nsc = sum(len(ws) for _, ws in items)
print(json.dumps(S, indent=1)); print(f"missing scores {missing}/{nsc}; empty raw {sum(not o for o in raw)}; api {apicache.STATS}")
tag = f"{'sonnet' if EXPL == 'sonnet' else EXPL}__{JUDGE}" + (f"_lim{LIMIT}" if LIMIT else "") + ("" if len(KINDS) == 4 else "_" + "-".join(KINDS))
fn = f"{REPO}/notes/api_t3_{tag}.json"
assert not os.path.exists(fn), f"{fn} exists; not overwriting"
json.dump(dict(explainer=EXPL, judge=JUDGE, kinds=KINDS, n_scores=nsc, missing_scores=missing, empty_raw=sum(not o for o in raw),
               summary=S, neurons=out), open(fn, "w"), indent=1)
print("wrote", fn)
