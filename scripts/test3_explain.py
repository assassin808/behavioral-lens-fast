"""Test 3b/3c: exemplars + auto-interp with detection scoring for Astra's 128 neurons (+16 random neurons per layer as reference).

Protocol adapted from /scratch/yang0531/bx/src/neuron_explain.py (readable neuron lens):
  windows   : from scripts/test3_collect.py; a neuron's "activation" = its signed write along u, c_j = act_j * (u^T w_j) (base weights).
              Top windows over all contexts (loop / normal math, IFEval, MMLU-Pro), at most 2 per context, near-duplicates removed
              (same 7-token context around the peak), 33 tokens (24 before, 8 after the peak).
  explain   : 8 top windows (top[0::2]) with the peak and tokens > 0.6*max marked, the source + whether the token is inside a
              verbatim-repeated passage, and the neuron's promoted tokens (logit lens of w_j) -> one-sentence description.
  detect    : held-out positives (top[1::2][:8], disjoint windows, checked) vs two negative sets of random tokens:
              'matched' = same source and same loop status as each positive (bx protocol: tests specificity beyond the segment);
              'nonloop' = same source, outside repeated passages (tests whether the neuron is more than "inside a loop").
              The simulator gives P(Yes) for "would this neuron fire strongly on the marked token?"; balanced accuracy and AP.
              Controls: shuffled description (another neuron's), segment-only description (the neuron's main segment).
Explainer backend is swappable: EXPLAINER=vllm:<hf model> (default vllm:Qwen/Qwen3-14B) or a class implementing
generate(prompts, max_tokens) -> list[str] and p_yes(prompts) -> list[float] (see Explainer below; an API backend can be added).
    python scripts/test3_explain.py [MAXT]    out: notes/test3_autointerp_{MAXT}_{explainer}.json, notes/test3_exemplars_{MAXT}.md
"""
import os, sys, json, math, random, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODELS, REPO, DATA_OUT

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
EXPL = os.environ.get("EXPLAINER", "vllm:Qwen/Qwen3-14B")
STAGE = os.environ.get("STAGE", "all")          # 'prep' = exemplars only (CPU), 'all' = + explainer
D = torch.load(f"{DATA_OUT}/t3_collect_{MAXT}{os.environ.get('SUFFIX', '')}.pt", weights_only=False)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{MODELS}/base")
meta = D["meta"]
SRCNAME = {"gsm8k": "GSM8K math reasoning", "math": "competition-math reasoning", "ifeval": "instruction-following task",
           "mmlupro": "multiple-choice question (MMLU-Pro)"}
CTX = ("We study a small language model (Qwen3.5-0.8B) writing long step-by-step reasoning for math problems, instruction-following "
       "tasks and multiple-choice questions. It often gets stuck repeating an earlier passage verbatim (a generation loop); each "
       "excerpt says whether the marked token lies inside such a verbatim-repeated passage.")
_dec = {}
def piece(i):
    if i not in _dec: _dec[i] = tok.decode([i])
    return _dec[i]
def seg(w): return f"{SRCNAME[meta[w['ctx']]['src']]}; inside repeated passage: {'yes' if w['in_loop'] else 'no'}"
def segkey(w): return (meta[w["ctx"]]["src"], bool(w["in_loop"]))
def render(w, acts, mark=True, peak_only=False):
    mx = max(max(acts), 1e-6); out = []
    for i, (t, a) in enumerate(zip(w["ids"], acts)):
        t = piece(t).replace("\n", "\\n")
        out.append(f"[[{t}]]" if mark and (i == w["peak"] or (not peak_only and a > 0.6 * mx)) else t)
    return "".join(out)
def wkey(w): return tuple(w["ids"][max(0, w["peak"] - 3): w["peak"] + 4])

# ---------------- promoted tokens (logit lens of base w_j through the final norm and tied embedding)
from safetensors import safe_open
f = safe_open(f"{MODELS}/base/model.safetensors-00001-of-00001.safetensors", "pt")
E = f.get_tensor("model.language_model.embed_tokens.weight").float(); nw = f.get_tensor("model.language_model.norm.weight").float()
gain = (1 + nw) if nw.abs().mean() < 0.5 else nw          # zero-centred RMSNorm weight in Qwen3.5 -> (1 + w)
W = {L: f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").float() for L in [15, 19]}

# ---------------- per-neuron windows
byn = collections.defaultdict(list)
for w in D["windows"]: byn[(w["layer"], w["unit"])].append(w)
rw = D["rwindows"]; rw_by_seg = collections.defaultdict(list)
for i, w in enumerate(rw): rw_by_seg[(meta[w["ctx"]]["src"], bool(w["in_loop"]))].append(i)
N = []
for L in [15, 19]:
    track = list(D["track"][L]); cols = set(map(int, D["cols"][L]))
    for k, j in enumerate(track):
        ws = sorted(byn[(L, int(j))], key=lambda w: -w["val"]); top, seen, per_ctx = [], set(), collections.Counter()
        for w in ws:
            if wkey(w) in seen or per_ctx[w["ctx"]] >= 2: continue
            seen.add(wkey(w)); per_ctx[w["ctx"]] += 1; top.append(w)
            if len(top) >= 16: break
        n = dict(layer=L, unit=int(j), k=k, is_random=int(j) not in cols, top=top)
        logits = E @ (W[L][:, j] * gain); n["promotes"] = [piece(i) for i in logits.topk(10).indices.tolist()]
        n["suppresses"] = [piece(i) for i in (-logits).topk(10).indices.tolist()]
        n["top_loop_share"] = float(np.mean([w["in_loop"] for w in top])) if top else None
        n["top_src"] = dict(collections.Counter(meta[w["ctx"]]["src"] for w in top))
        ex, pos = top[0::2][:8], top[1::2][:8]
        exk = {wkey(w) for w in ex}; pos = [w for w in pos if wkey(w) not in exk]            # held-out check: no near-duplicates
        R = random.Random(L * 10000 + int(j)); negm, negn = [], []
        used = set()
        for w in pos:
            for target, lst in [(segkey(w), negm), ((meta[w["ctx"]]["src"], False), negn)]:
                pool = [i for i in rw_by_seg.get(target, []) if i not in used]
                if pool: i = R.choice(pool); used.add(i); lst.append(rw[i])
        n.update(_ex=ex, _pos=pos, _negm=negm, _negn=negn)
        sk = collections.Counter(segkey(w) for w in top).most_common(1)
        n["seg_main"] = sk[0][0] if sk else ("math", False)
        N.append(n)
def ract(w, n):  # acts of neuron n in a random window
    return [float(x) for x in w["acts"][n["layer"]][:, n["k"]]]

# exemplar dump (human-readable)
md = [f"# Test 3b — top exemplars of Astra's 128 neurons (base model, c_j = act_j·uᵀw_j, {len(meta)} contexts, ≤{MAXT} tokens)\n",
      "Auto-generated by `scripts/test3_explain.py`. Peak token and tokens > 0.6·max marked [[like this]]. 'loop' = inside a verbatim-repeated "
      "passage (32-gram seen before). Top windows: ≤2 per context, near-duplicates removed.\n"]
for n in N:
    if n["is_random"]: continue
    md.append(f"\n## L{n['layer']}.{n['unit']}  top-16 loop share {n['top_loop_share']:.2f}, sources {n['top_src']}\npromotes: {n['promotes'][:8]}\n")
    for w in n["top"][:4]:
        md.append(f"- ({meta[w['ctx']]['src']}, {'loop' if w['in_loop'] else 'non-loop'}, c={w['val']:.3f}) `{render(w, w['acts'])}`")
open(f"{REPO}/notes/test3_exemplars_{MAXT}.md", "w").write("\n".join(md) + "\n")
summ_ex = {grp: dict(n=len(xs), top16_loop_share=float(np.mean([x["top_loop_share"] for x in xs])),
                     top16_src=dict(sum((collections.Counter(x["top_src"]) for x in xs), collections.Counter())))
           for grp, xs in [("astra_L15", [n for n in N if not n["is_random"] and n["layer"] == 15]),
                           ("astra_L19", [n for n in N if not n["is_random"] and n["layer"] == 19]),
                           ("random", [n for n in N if n["is_random"]])]}
print(json.dumps(summ_ex, indent=1), flush=True)
if STAGE == "prep":
    json.dump(dict(exemplar_summary=summ_ex), open(f"{REPO}/notes/test3_exemplar_summary_{MAXT}.json", "w"), indent=1); sys.exit()


# ---------------- explainer backend
class VLLMExplainer:
    def __init__(self, name):
        from vllm import LLM
        self.llm = LLM(name, max_model_len=8192, gpu_memory_utilization=0.88, seed=0); self.tok = self.llm.get_tokenizer()
        self.yes = [self.tok.encode(x, add_special_tokens=False)[0] for x in ["Yes", " Yes"]]
        self.no = [self.tok.encode(x, add_special_tokens=False)[0] for x in ["No", " No"]]
    def chat(self, p): return self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    def generate(self, ps, max_tokens=60):
        from vllm import SamplingParams
        return [o.outputs[0].text.strip().split("\n")[0] for o in self.llm.generate([self.chat(p) for p in ps], SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)]
    def p_yes(self, ps):
        from vllm import SamplingParams
        outs = self.llm.generate([self.chat(p) for p in ps], SamplingParams(temperature=0, max_tokens=1, logprobs=20), use_tqdm=False); r = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            y = sum(math.exp(lp[i].logprob) for i in self.yes if i in lp); nn = sum(math.exp(lp[i].logprob) for i in self.no if i in lp)
            r.append(y / (y + nn) if y + nn > 0 else 0.5)
        return r
def make_explainer(spec):
    kind, name = spec.split(":", 1)
    if kind == "vllm": return VLLMExplainer(name)
    raise NotImplementedError(f"explainer backend {kind!r}: add a class with generate() and p_yes() (e.g. an API client)")
X = make_explainer(EXPL)

def explain_prompt(n):
    ex = "\n".join(f"{k + 1}. ({seg(w)}) ...{render(w, w['acts'])}..." for k, w in enumerate(n["_ex"]))
    return (f"{CTX}\nBelow are text excerpts where one neuron of the model fires most strongly. Tokens where it fires are marked [[like this]].\n"
            f"{ex}\n\nWhen this neuron fires, it pushes the model to output tokens like: {', '.join(repr(t) for t in n['promotes'][:10])}\n\n"
            "In ONE short sentence (max 20 words), describe what this neuron detects or does. Be specific about the textual pattern "
            "(e.g. which token, in which kind of text, in what situation). Answer with just the sentence.")
def detect_prompt(desc, w, acts):
    return (f"{CTX}\nA neuron of this model is described as: \"{desc}\"\n\nText excerpt ({seg(w)}), one token marked [[like this]]:\n"
            f"...{render(w, acts, peak_only=True)}...\n\nWould this neuron fire strongly on the marked token? Answer Yes or No.")

ok = [n for n in N if len(n["_ex"]) >= 4 and len(n["_pos"]) >= 4 and len(n["_negm"]) >= 4]
descs = X.generate([explain_prompt(n) for n in ok])
for n, s in zip(ok, descs): n["desc"] = s
perm = list(range(len(ok))); random.Random(0).shuffle(perm)
for i, n in enumerate(ok):
    n["desc_shuf"] = ok[perm[i] if perm[i] != i else (i + 1) % len(ok)]["desc"]
    s, lp = n["seg_main"]
    n["desc_seg"] = (f"Fires on tokens inside a passage that the model is repeating verbatim (a generation loop), in {SRCNAME[s]} text." if lp
                     else f"Fires on tokens in {SRCNAME[s]} text.")
    n["desc_loop"] = "Fires on tokens inside a passage that the model is repeating verbatim (a generation loop)."
KINDS = ["desc", "desc_shuf", "desc_seg", "desc_loop"]
jobs = []
for i, n in enumerate(ok):
    for kind in KINDS:
        for w in n["_pos"]: jobs.append((i, kind, "pos", detect_prompt(n[kind], w, w["acts"])))
        for nk in ["_negm", "_negn"]:
            for w in n[nk]: jobs.append((i, kind, nk, detect_prompt(n[kind], w, ract(w, n))))
ps = X.p_yes([j[3] for j in jobs])
sc = collections.defaultdict(list)
for (i, kind, lab, _), p in zip(jobs, ps): sc[(i, kind, lab)].append(p)
from sklearn.metrics import average_precision_score
def score(pos, neg):
    if len(neg) < 2: return None, None
    tpr = np.mean([p > .5 for p in pos]); tnr = np.mean([p <= .5 for p in neg])
    return float((tpr + tnr) / 2), float(average_precision_score([1] * len(pos) + [0] * len(neg), pos + neg))
out = []
for i, n in enumerate(ok):
    r = {k: v for k, v in n.items() if not k.startswith("_") and k not in ("top",)}
    for kind in KINDS:
        for nk in ["_negm", "_negn"]:
            r[f"bacc_{kind}{nk}"], r[f"ap_{kind}{nk}"] = score(sc[(i, kind, "pos")], sc[(i, kind, nk)])
    r["n_pos"], r["n_negm"], r["n_negn"] = len(n["_pos"]), len(n["_negm"]), len(n["_negn"])
    r["examples"] = [render(w, w["acts"]) for w in n["_ex"][:3]]
    out.append(r)
def summ(xs):
    s = dict(n=len(xs))
    for kind in KINDS:
        for nk in ["_negm", "_negn"]:
            v = [x[f"ap_{kind}{nk}"] for x in xs if x[f"ap_{kind}{nk}"] is not None]
            s[f"ap_{kind}{nk}"] = float(np.mean(v)) if v else None; s[f"n_{nk}"] = len(v)
    s["frac_ap_desc_negm_gt_shuf"] = float(np.mean([x["ap_desc_negm"] > x["ap_desc_shuf_negm"] for x in xs if x["ap_desc_negm"] is not None]))
    return s
S = {g: summ([x for x in out if f(x)]) for g, f in [("astra_L15", lambda x: not x["is_random"] and x["layer"] == 15),
                                                     ("astra_L19", lambda x: not x["is_random"] and x["layer"] == 19),
                                                     ("random", lambda x: x["is_random"])]}
print(json.dumps(S, indent=1), flush=True)
for x in out[:20] + [x for x in out if x["is_random"]][:4]:
    print(f"L{x['layer']}.{x['unit']} rand={x['is_random']} ap_m={x['ap_desc_negm']} ap_nl={x['ap_desc_negn']} seg_nl={x['ap_desc_seg_negn']} | {x['desc']}", flush=True)
tag = EXPL.split("/")[-1]
json.dump(dict(explainer=EXPL, MAXT=MAXT, n_ctx=len(meta), exemplar_summary=summ_ex, summary=S, neurons=out),
          open(f"{REPO}/notes/test3_autointerp_{MAXT}_{tag}.json", "w"), indent=1)
