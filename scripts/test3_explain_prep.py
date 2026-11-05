"""Test 3b prep (run where the collected tensors are; no explainer): per-neuron top windows (exemplars) and held-out detection
sets for Astra's 128 neurons (+16 random neurons per layer as reference), rendered as text, plus the human-readable exemplar note.

  windows : from scripts/test3_collect.py; a neuron's "activation" = its signed write along u, c_j = act_j * (u^T w_j) (base weights).
            Top windows over all contexts (loop / normal math, IFEval, MMLU-Pro), at most 2 per context, near-duplicates removed
            (same 7-token context around the peak), 33 tokens (24 before, 8 after the peak).
  splits  : explanation set = top[0::2][:8]; held-out positives = top[1::2][:8] minus any whose 7-token peak context occurs in the
            explanation set; negatives = random tokens, 'matched' (same source + same loop status as each positive, bx protocol) and
            'nonloop' (same source, outside repeated passages), drawn without replacement.
    python scripts/test3_explain_prep.py [MAXT]   out: $T38_DATA/t3_explain_prep_{MAXT}.json, notes/test3_exemplars_{MAXT}.md,
                                                       notes/test3_exemplar_summary_{MAXT}.json
Then scripts/test3_explain.py (explainer + detection scoring; can run on the login node with an API backend).
"""
import os, sys, json, math, random, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODEL_DIR, RUNTAG, weights_file, REPO, DATA_OUT

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
D = torch.load(f"{DATA_OUT}/t3_collect_{MAXT}{os.environ.get('SUFFIX', '')}.pt", weights_only=False)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR["base"])
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
f = safe_open(weights_file("base"), "pt")
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
      "Auto-generated by `scripts/test3_explain_prep.py`. Peak token and tokens > 0.6·max marked [[like this]]. 'loop' = inside a verbatim-repeated "
      "passage (32-gram seen before). Top windows: ≤2 per context, near-duplicates removed.\n"]
for n in N:
    if n["is_random"]: continue
    md.append(f"\n## L{n['layer']}.{n['unit']}  top-16 loop share {n['top_loop_share']:.2f}, sources {n['top_src']}\npromotes: {n['promotes'][:8]}\n")
    for w in n["top"][:4]:
        md.append(f"- ({meta[w['ctx']]['src']}, {'loop' if w['in_loop'] else 'non-loop'}, c={w['val']:.3f}) `{render(w, w['acts'])}`")
open(f"{REPO}/notes/test3_exemplars_{MAXT}_{RUNTAG}.md", "w").write("\n".join(md) + "\n")
summ_ex = {grp: dict(n=len(xs), top16_loop_share=float(np.mean([x["top_loop_share"] for x in xs])),
                     top16_src=dict(sum((collections.Counter(x["top_src"]) for x in xs), collections.Counter())))
           for grp, xs in [("astra_L15", [n for n in N if not n["is_random"] and n["layer"] == 15]),
                           ("astra_L19", [n for n in N if not n["is_random"] and n["layer"] == 19]),
                           ("random", [n for n in N if n["is_random"]])]}
print(json.dumps(summ_ex, indent=1), flush=True)
json.dump(dict(exemplar_summary=summ_ex), open(f"{REPO}/notes/test3_exemplar_summary_{MAXT}_{RUNTAG}.json", "w"), indent=1)
def pack(w, acts, peak_only=False): return dict(seg=seg(w), segkey=list(segkey(w)), text=render(w, acts, peak_only=peak_only), ctx=w["ctx"], pos=int(w["pos"]))
P_ = []
for n in N:
    P_.append(dict(layer=n["layer"], unit=n["unit"], is_random=n["is_random"], promotes=n["promotes"], suppresses=n["suppresses"],
                   top_loop_share=n["top_loop_share"], top_src=n["top_src"], seg_main=list(n["seg_main"]),
                   ex=[pack(w, w["acts"]) for w in n["_ex"]], pos=[pack(w, w["acts"], True) for w in n["_pos"]],
                   negm=[pack(w, ract(w, n), True) for w in n["_negm"]], negn=[pack(w, ract(w, n), True) for w in n["_negn"]],
                   top_vals=[float(w["val"]) for w in n["top"]]))
chk = dict(n_neurons=len(P_), ex_pos_same_ctx_pos=sum(len({(x["ctx"], x["pos"]) for x in p["ex"]} & {(x["ctx"], x["pos"]) for x in p["pos"]}) for p in P_))
print("split check", chk)
json.dump(dict(MAXT=MAXT, n_ctx=len(meta), split_check=chk, exemplar_summary=summ_ex, neurons=P_),
          open(f"{DATA_OUT}/t3_explain_prep_{MAXT}.json", "w"))
print("wrote prep", len(P_))
