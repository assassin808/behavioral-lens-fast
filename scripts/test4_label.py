"""Test 4: auto-interp labels + detection scores for the selected SAE features (run on the LOGIN node; needs internet).

    set -a; . ~/.env; set +a; python3 scripts/test4_label.py ANALYZE.json OUT.json [--model claude-sonnet-5-5] [--subset N]

Label: the explainer sees 12 top-activating train contexts (activating token in [[ ]], activation value) and writes a
short label + one-sentence description. Score (detection, held-out test split): the scorer sees the label and 12 shuffled
contexts (6 activating sampled across activation quantiles, 6 random non-activating), token marked in [[ ]], and says which
ones activate; balanced accuracy is reported (0.5 = chance).
With --subset N (spot check) only the first N features (fixed seeded order) are labelled and scored.
"""
import os, sys, json, random, re, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
inp, outp = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
MODEL = args[args.index("--model") + 1] if "--model" in args else "claude-sonnet-5-5"
SUB = int(args[args.index("--subset") + 1]) if "--subset" in args else None
assert not os.path.exists(outp), "refusing to overwrite"
B, KEY = os.environ["VECTOR_BASE_URL"].rstrip("/"), os.environ["VECTOR_API_KEY"]
R = json.load(open(inp))


def chat(prompt, max_tokens=400):
    for a in range(5):
        try:
            req = urllib.request.Request(B + "/chat/completions", data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                         "max_tokens": max_tokens}).encode(),   # temperature is rejected for these models
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(5 * (a + 1)); err = str(e)[:200]
    return "ERROR " + err


def show(e, act=True):
    c = (e["pre"][-220:] + "[[" + e["tok"] + "]]" + e["post"]).replace("\n", "\\n")
    return c + (f"   (activation {e['act']:.2f})" if act and "act" in e else "")


def label(nm, j, o):
    ex = o["top"][:12]
    p = ("You are interpreting one latent of a sparse autoencoder trained on the residual stream of a small language model "
         "(Qwen3.5-0.8B). The texts are mostly the model's own long chain-of-thought for math / instruction-following problems "
         "(some of which degenerate into verbatim repetition loops) plus Wikipedia. Below are the contexts where the latent is "
         "most active; the token where it fires is marked [[like this]] (the latent reads the residual at that token, i.e. it can "
         "reflect the context so far and the model's prediction of what comes next).\n\n" +
         "\n".join(f"{i + 1}. {show(e)}" for i, e in enumerate(ex)) +
         "\n\nAnswer in exactly this format:\nLABEL: <3-8 word label>\nDESCRIPTION: <one sentence>\nREPETITION_RELATED: <yes/no — is it about the text "
         "repeating itself verbatim, a repetition loop, or re-checking / restating earlier content?>")
    t = chat(p)
    g = lambda k: (re.search(k + r":\s*(.*)", t) or [None, ""])[1].strip()
    return dict(label=g("LABEL"), description=g("DESCRIPTION"), repetition_related=g("REPETITION_RELATED").lower().startswith("y"), raw=t)


def score(lab, o, seed):
    pos, neg = o.get("score_pos", []), o.get("score_neg", [])
    if len(pos) < 3: return None
    allx = [(e, 1) for e in pos] + [(e, 0) for e in neg]; random.Random(seed).shuffle(allx)
    p = ("A sparse-autoencoder latent of a language model has this explanation:\n" + f"LABEL: {lab['label']}\nDESCRIPTION: {lab['description']}\n\n"
         "For each context below, the token of interest is marked [[like this]]. Decide for which contexts the latent is ACTIVE at the "
         "marked token. About half are active.\n\n" + "\n".join(f"{i + 1}. {show(e, act=False)}" for i, (e, _) in enumerate(allx)) +
         "\n\nRespond with ONLY one line and no reasoning: ACTIVE: <comma-separated numbers>")
    t = chat(p, 800)
    ms = re.findall(r"ACTIVE:\s*([\d,\s]*)", t); pred = set(int(x) for x in re.findall(r"\d+", ms[-1])) if ms else set()
    y = [l for _, l in allx]; yh = [1 if i + 1 in pred else 0 for i in range(len(allx))]
    tpr = sum(a and b for a, b in zip(y, yh)) / max(1, sum(y)); tnr = sum((not a) and (not b) for a, b in zip(y, yh)) / max(1, len(y) - sum(y))
    return dict(bal_acc=(tpr + tnr) / 2, tpr=tpr, tnr=tnr, raw=t)


jobs = []
for nm in ["ours16", "pub18", "ours20"]:
    for j, o in R[nm]["exemplars"].items():
        if o.get("top"): jobs.append((nm, j, o))
if SUB:
    random.Random(0).shuffle(jobs); jobs = jobs[:SUB]
print("features to label", len(jobs), "model", MODEL, flush=True)


def work(a):
    nm, j, o = a
    lab = label(nm, j, o); sc = score(lab, o, zlib.crc32(f"{nm}:{j}".encode()))
    return f"{nm}:{j}", dict(lab, score=sc)


out = {}
with ThreadPoolExecutor(4) as ex:
    for k, v in ex.map(work, jobs):
        out[k] = v; print(k, "|", v["label"], "| rep", v["repetition_related"], "| bal_acc", v["score"] and round(v["score"]["bal_acc"], 2), flush=True)
json.dump(dict(model=MODEL, n=len(out), labels=out), open(outp, "w"), indent=1)
