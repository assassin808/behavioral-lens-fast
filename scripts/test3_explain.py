"""Test 3c: auto-interp with detection scoring for Astra's 128 neurons (+16 random per layer), from scripts/test3_explain_prep.py.

Protocol adapted from /scratch/yang0531/bx/src/neuron_explain.py:
  explain : 8 top windows (peak + tokens > 0.6*max marked, source + loop status shown) + promoted tokens -> one sentence.
  detect  : held-out positives vs negatives ('matched' = same source + loop status; 'nonloop' = same source, not in a loop).
            Description kinds: the neuron's description, a shuffled one (another neuron's), segment-only (the neuron's main
            segment), loop-only ("fires inside a verbatim-repeated passage"). Balanced accuracy at 0.5 and AP.
Backends (EXPLAINER env):
  api:<model>   OpenAI-compatible chat API (VECTOR_BASE_URL / VECTOR_API_KEY from ~/.env; default api:claude-sonnet-5-5).
                No logprobs, so detection is done the batched way (as in EleutherAI's sae-auto-interp detection scorer): one
                call per (neuron, description) with all of that neuron's held-out positives and negatives shuffled together,
                the judge returns a 0-100 score per excerpt -> score/100 is used as P(fires).
  vllm:<model>  local model; per-excerpt P(Yes) from first-token logprobs (bx protocol), e.g. vllm:Qwen/Qwen3-14B.
Any backend = a class with generate(prompts) -> list[str] and score_batch(list of (desc, excerpts)) -> list[list[float]].
    python scripts/test3_explain.py [MAXT] [subset]    out: notes/test3_autointerp_{MAXT}_{model}.json
subset: 'all' (default) or an int k = only the first k Astra neurons per layer + 4 random per layer (spot checks).
"""
import os, sys, json, math, random, re, collections, concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import REPO, DATA_OUT, RUNTAG

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "all"
EXPL = os.environ.get("EXPLAINER", "api:claude-sonnet-5-5")
PR = json.load(open(f"{DATA_OUT}/t3_explain_prep_{MAXT}.json"))
N = PR["neurons"]
if SUBSET != "all":
    k = int(SUBSET); N = [n for L in [15, 19] for n in ([x for x in N if x["layer"] == L and not x["is_random"]][:k] +
                                                       [x for x in N if x["layer"] == L and x["is_random"]][:4])]
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


def detect_one(desc, w):
    return (f"{CTX}\nA neuron of this model is described as: \"{desc}\"\n\nText excerpt ({w['seg']}), one token marked [[like this]]:\n"
            f"...{w['text']}...\n\nWould this neuron fire strongly on the marked token? Answer Yes or No.")


def detect_batch(desc, ws):
    ex = "\n".join(f"{k + 1}. ({w['seg']}) ...{w['text']}..." for k, w in enumerate(ws))
    return (f"{CTX}\nA neuron of this model is described as: \"{desc}\"\n\nBelow are {len(ws)} text excerpts, each with ONE token marked "
            f"[[like this]]. Some are tokens where the neuron fires strongly, others are random tokens.\n{ex}\n\n"
            "For each excerpt, rate from 0 to 100 how likely it is that the neuron fires strongly on the marked token, judging only from "
            f"the description. Answer with exactly {len(ws)} lines of the form `<index>: <score>` and nothing else.")


class APIExplainer:
    def __init__(self, model):
        from openai import OpenAI
        self.c = OpenAI(base_url=os.environ["VECTOR_BASE_URL"], api_key=os.environ["VECTOR_API_KEY"], max_retries=6, timeout=180)
        self.model = model; self.failed = 0
    def _call(self, p, max_tokens):
        for _ in range(4):
            try:
                r = self.c.chat.completions.create(model=self.model, messages=[{"role": "user", "content": p}], max_tokens=max_tokens)
                return r.choices[0].message.content or ""
            except Exception as e:
                print("api error", type(e).__name__, str(e)[:200], flush=True)
        self.failed += 1
        if self.failed > 20: raise RuntimeError("too many API failures; not writing results")
        return ""
    def _map(self, ps, max_tokens):
        with cf.ThreadPoolExecutor(int(os.environ.get("API_THREADS", 12))) as ex:
            return list(ex.map(lambda p: self._call(p, max_tokens), ps))
    def generate(self, ps): return [o.strip().split("\n")[0] for o in self._map(ps, 80)]
    def score_batch(self, items):
        outs = self._map([detect_batch(d, ws) for d, ws in items], 400); res = []
        for (d, ws), o in zip(items, outs):
            sc = {int(a): float(b) for a, b in re.findall(r"^\s*(\d+)\s*[:.)-]\s*(\d+(?:\.\d+)?)", o, re.M)}
            res.append([min(max(sc.get(k + 1, 50.0), 0), 100) / 100 for k in range(len(ws))])
            if len(sc) < len(ws): print(f"parse: {len(sc)}/{len(ws)} scores", flush=True)
        return res


class VLLMExplainer:
    def __init__(self, name):
        from vllm import LLM
        self.llm = LLM(name, max_model_len=8192, gpu_memory_utilization=0.88, seed=0); self.tok = self.llm.get_tokenizer()
        self.yes = [self.tok.encode(x, add_special_tokens=False)[0] for x in ["Yes", " Yes"]]
        self.no = [self.tok.encode(x, add_special_tokens=False)[0] for x in ["No", " No"]]
    def chat(self, p): return self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    def generate(self, ps):
        from vllm import SamplingParams
        return [o.outputs[0].text.strip().split("\n")[0] for o in self.llm.generate([self.chat(p) for p in ps], SamplingParams(temperature=0, max_tokens=60), use_tqdm=False)]
    def score_batch(self, items):
        from vllm import SamplingParams
        flat = [detect_one(d, w) for d, ws in items for w in ws]
        outs = self.llm.generate([self.chat(p) for p in flat], SamplingParams(temperature=0, max_tokens=1, logprobs=20), use_tqdm=False); r = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            y = sum(math.exp(lp[i].logprob) for i in self.yes if i in lp); nn = sum(math.exp(lp[i].logprob) for i in self.no if i in lp)
            r.append(y / (y + nn) if y + nn > 0 else 0.5)
        it = iter(r); return [[next(it) for _ in ws] for _, ws in items]


kind, name = EXPL.split(":", 1)
X = {"api": APIExplainer, "vllm": VLLMExplainer}[kind](name)

ok = [n for n in N if len(n["ex"]) >= 4 and len(n["pos"]) >= 4 and len(n["negm"]) >= 4]
print(f"{len(ok)} neurons with enough windows (of {len(N)}); explainer {EXPL}", flush=True)
for n, s in zip(ok, X.generate([explain_prompt(n) for n in ok])): n["desc"] = s
perm = list(range(len(ok))); random.Random(0).shuffle(perm)
for i, n in enumerate(ok):
    n["desc_shuf"] = ok[perm[i] if perm[i] != i else (i + 1) % len(ok)]["desc"]
    s, lp = n["seg_main"]
    n["desc_seg"] = (f"Fires on tokens inside a passage that the model is repeating verbatim (a generation loop), in {SRCNAME[s]} text." if lp
                     else f"Fires on tokens in {SRCNAME[s]} text.")
    n["desc_loop"] = "Fires on tokens inside a passage that the model is repeating verbatim (a generation loop)."
KINDS = ["desc", "desc_shuf", "desc_seg", "desc_loop"]
items, keys = [], []
for i, n in enumerate(ok):
    ws = [(1, "pos", w) for w in n["pos"]] + [(0, "negm", w) for w in n["negm"]] + [(0, "negn", w) for w in n["negn"]]
    random.Random(1000 + i).shuffle(ws)
    for kd in KINDS:
        items.append((n[kd], [w for _, _, w in ws])); keys.append((i, kd, [(l, g) for l, g, _ in ws]))
scores = X.score_batch(items)
from sklearn.metrics import average_precision_score
def score(pos, neg):
    if len(neg) < 2 or len(pos) < 2: return None, None
    tpr = np.mean([p > .5 for p in pos]); tnr = np.mean([p <= .5 for p in neg])
    return float((tpr + tnr) / 2), float(average_precision_score([1] * len(pos) + [0] * len(neg), pos + neg))
out = [{k: v for k, v in n.items() if k not in ("ex", "pos", "negm", "negn")} | dict(examples=[w["text"] for w in n["ex"][:3]],
       n_pos=len(n["pos"]), n_negm=len(n["negm"]), n_negn=len(n["negn"])) for n in ok]
for (i, kd, lab), sc in zip(keys, scores):
    pos = [s for (l, g), s in zip(lab, sc) if g == "pos"]
    for g in ["negm", "negn"]:
        out[i][f"bacc_{kd}_{g}"], out[i][f"ap_{kd}_{g}"] = score(pos, [s for (l, gg), s in zip(lab, sc) if gg == g])
def summ(xs):
    s = dict(n=len(xs))
    for kd in KINDS:
        for g in ["negm", "negn"]:
            for m in ["ap", "bacc"]:
                v = [x[f"{m}_{kd}_{g}"] for x in xs if x.get(f"{m}_{kd}_{g}") is not None]
                s[f"{m}_{kd}_{g}"] = float(np.mean(v)) if v else None
    v = [x["ap_desc_negn"] - x["ap_desc_loop_negn"] for x in xs if x.get("ap_desc_negn") is not None and x.get("ap_desc_loop_negn") is not None]
    s["mean_ap_desc_minus_loop_negn"] = float(np.mean(v)) if v else None
    return s
S = {g: summ([x for x in out if f(x)]) for g, f in [("astra_L15", lambda x: not x["is_random"] and x["layer"] == 15),
                                                     ("astra_L19", lambda x: not x["is_random"] and x["layer"] == 19),
                                                     ("random", lambda x: x["is_random"])]}
print(json.dumps(S, indent=1), flush=True)
for x in out[:12] + [x for x in out if x["is_random"]][:4]:
    print(f"L{x['layer']}.{x['unit']} rand={x['is_random']} ap_m={x['ap_desc_negm']} ap_nl={x['ap_desc_negn']} loop_nl={x['ap_desc_loop_negn']} | {x['desc']}", flush=True)
tag = name.split("/")[-1] + ("" if SUBSET == "all" else f"_sub{SUBSET}") + f"_{RUNTAG}"
json.dump(dict(explainer=EXPL, MAXT=MAXT, n_ctx=PR["n_ctx"], split_check=PR["split_check"], exemplar_summary=PR["exemplar_summary"],
               scoring="batched 0-100 scores per (neuron, description)" if kind == "api" else "per-excerpt P(Yes) logprobs",
               summary=S, neurons=out), open(f"{REPO}/notes/test3_autointerp_{MAXT}_{tag}.json", "w"), indent=1)
print("wrote", f"notes/test3_autointerp_{MAXT}_{tag}.json")
