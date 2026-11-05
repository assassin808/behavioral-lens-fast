"""Shared helpers for tests 2 and 4 (branch yang/t2-t4): corpus, prompts, loop mask, models, u, token classes.

Corpus / prompt / loop-mask logic follows scripts/q2_neuron_activations.py and yang/t3-t8's test3_common.py (same rules,
so results are comparable across tests).
Model dirs: /scratch/yang0531/blf_models/{base,p50c}
  base = Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17
  p50c = base + astra_logs/exp8_math/runs/p50c/edit via astra_logs/exp8_math/code/apply_edit.py (sha256-checked)
"""
import glob, gzip, json, os, random
import numpy as np

MODELS = os.environ.get("BLF_MODELS", "/scratch/yang0531/blf_models")
EXT = f"{MODELS}/extdata"
DATA = os.environ.get("BLF_DATA", "/scratch/yang0531/blf_data")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
DSETS = ["gsm8k", "math500", "ifeval", "mmlupro1k"]


def loop_mask(ids, n=32):
    """token i is 'in loop' if the n-gram ending at i already occurred earlier (same rule as q2)."""
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m


_Q = None
def question(id_):
    global _Q
    if _Q is None:
        q = {}
        for l in open(f"{REPO}/data/gsm8k_test.jsonl"):
            r = json.loads(l); q[r["id"]] = r["question"] + "\n" + INSTR
        for l in open(f"{REPO}/data/math500.jsonl"):
            r = json.loads(l); q[r["id"]] = r["problem"] + "\n" + INSTR
        for l in open(f"{EXT}/ifeval.jsonl"):
            r = json.loads(l); q[r["id"]] = r["prompt"]
        for l in open(f"{EXT}/mmlupro.jsonl"):
            r = json.loads(l)
            opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
            q[r["id"]] = (f"The following is a multiple choice question about {r['category']}. Think step by step and then finish your "
                          f"answer with \"the answer is (X)\" where X is the correct letter choice.\n\nQuestion: {r['question']}\nOptions:\n{opts}")
        _Q = q
    return _Q.get(id_)


def astra_rows(tag, ds):
    rows = []
    for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/{tag}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
    for r in rows: r["ds"] = ds; r["gen_by"] = tag
    return rows


def encode(tok, id_, text, maxt=16384):
    q = question(id_)
    if q is None: return None, None
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; gid = tok(text, add_special_tokens=False).input_ids
    return (pid + gid)[:maxt], len(pid)


def split_of(id_):
    """deterministic train/held-out split by problem id (same id -> same split for both generators). 15 % held out."""
    import hashlib
    return "test" if int(hashlib.md5(id_.encode()).hexdigest(), 16) % 100 < 15 else "train"


def chat_items(tok, maxt=16384, tags=("unedited", "p50c"), dsets=DSETS):
    """all Astra generations with known prompt: dicts id, ds, gen_by, ids (np.int32), P, loop (bool mask, prompt excluded), split."""
    out = []
    for tag in tags:
        for ds in dsets:
            for r in astra_rows(tag, ds):
                ids, P = encode(tok, r["id"], r["text"], maxt)
                if ids is None: continue
                m = loop_mask(ids); m[:P] = False
                out.append(dict(id=r["id"], ds=ds, gen_by=tag, ids=np.array(ids, np.int32), P=P, loop=m,
                                finished="</think>" in r["text"], split=split_of(r["id"])))
    return out


def wiki_docs(n_tokens, tok, seq=1024, seed=0, skip_docs=0):
    """WikiText-103 train articles concatenated into `seq`-token chunks (BOS-free; Qwen has no BOS). Returns list of np arrays."""
    import pyarrow.parquet as pq
    t = pq.read_table(f"{DATA}/wikitext/train0.parquet").column("text").to_pylist()
    arts, cur = [], []
    for line in t:
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(line)
    arts = [a for a in arts if len(a) > 2000][skip_docs:]
    out, tot = [], 0
    for a in arts:
        ids = tok(a, add_special_tokens=False).input_ids
        for i in range(0, len(ids) - seq + 1, seq):
            out.append(np.array(ids[i:i + seq], np.int32)); tot += seq
            if tot >= n_tokens: return out
    return out


def load_model(tag, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = f"{MODELS}/{ 'base' if tag in ('base', 'unedited') else tag}"
    tok = AutoTokenizer.from_pretrained(f"{MODELS}/base")
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to(device).eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m, tok


def lm_parts(m):
    """(decoder layers, final norm, lm_head) of the Qwen3.5 HF model."""
    lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
    return lm, lm.layers, lm.norm, m.lm_head


def get_u():
    return {15: np.load(f"{REPO}/shared/u_L15.npy").astype(np.float32), 19: np.load(f"{REPO}/shared/u_L19.npy").astype(np.float32)}


CONCLUDE = [" So", " Therefore", " Thus", " Hence", " answer", " Answer", " result", " final", " Final", " boxed", "\\boxed",
            " total", " equals", " =", "</think>", " is", " conclusion", " done", " Done", " correct", "**", " The"]
CONTINUE = [" Wait", "Wait", " wait", " Let", " again", " check", " verify", " double", " Hmm", " But", " Alternatively",
            " re", " Maybe", " maybe", " Or", " Also", " Actually", " another", "\n", "\n\n", " -", " *", "  ", "    "]


def class_ids(tok, words):
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if len(t) == 1: ids.append(t[0])
    return sorted(set(ids))


def lens_contexts(tok, split="train", seed=0, n_loop=350, n_math=350, n_wiki=200, n_nonmath=100):
    """~1000 contexts for lens fitting/eval: (ids, sample positions, kind). Kinds: loop (unedited looping generations,
    <=8192 tok, 100 loop + 100 non-loop generated positions), math (finished, loop frac < 1 %, <=4096 tok),
    wiki (WikiText 1024-tok chunks), nonmath (finished IFEval / MMLU-Pro generations <=4096 tok)."""
    rng = np.random.default_rng(seed)
    items = [x for x in chat_items(tok, maxt=8192) if x["split"] == split]
    out = []
    def take(ids, cand, n):
        cand = np.asarray(cand)
        return np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False)) if len(cand) else np.array([], int)
    loopi = [x for x in items if x["gen_by"] == "unedited" and x["loop"].mean() > 0.03]
    for x in [loopi[i] for i in rng.permutation(len(loopi))[:n_loop]]:
        gen = np.arange(x["P"], len(x["ids"]))
        p = np.concatenate([take(x["ids"], np.flatnonzero(x["loop"]), 100), take(x["ids"], gen[~x["loop"][gen]], 100)])
        out.append((x["ids"], np.sort(p), "loop", x["loop"]))
    def short(x, mx=4096): return x["ids"][:mx], x["loop"][:mx]
    mathi = [x for x in items if x["ds"] in ("gsm8k", "math500") and x["finished"] and x["loop"].mean() < 0.01]
    for x in [mathi[i] for i in rng.permutation(len(mathi))[:n_math]]:
        ids, lp = short(x); out.append((ids, take(ids, np.arange(x["P"], len(ids)), 200), "math", lp))
    nm = [x for x in items if x["ds"] in ("ifeval", "mmlupro1k") and x["finished"] and x["loop"].mean() < 0.01]
    for x in [nm[i] for i in rng.permutation(len(nm))[:n_nonmath]]:
        ids, lp = short(x); out.append((ids, take(ids, np.arange(x["P"], len(ids)), 200), "nonmath", lp))
    wk = wiki_docs(1024 * (n_wiki + 50), tok, seq=1024, skip_docs=20000 if split == "train" else 25000)
    for w in wk[:n_wiki]:
        out.append((w, take(w, np.arange(16, len(w)), 200), "wiki", np.zeros(len(w), bool)))
    return out


class SAEWrap:
    """uniform interface: encode(raw residual [N,1024] fp32 cuda) -> [N, m]; dirs = unit decoder rows [m,1024]."""
    def __init__(self, kind, path, device="cuda"):
        import torch
        self.kind = kind
        if kind == "ours":
            ck = torch.load(path, map_location=device); st = ck["state"]; self.c = ck["scale"]
            self.W_enc, self.b_enc, self.W_dec, self.b_dec = st["W_enc"], st["b_enc"], st["W_dec"], st["b_dec"]
            self.thr = st["theta"] * torch.ones_like(self.b_enc); self.cfg = ck["cfg"]
        else:
            from safetensors.torch import load_file
            st = load_file(path, device=device); self.c = 1.0
            self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.thr = st["W_enc"], st["b_enc"], st["W_dec"], st["b_dec"], st["threshold"]
            self.cfg = {}
        self.dirs = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.m = self.W_dec.shape[0]

    def pre(self, x, sel=None):
        x = x / self.c
        W, b = (self.W_enc, self.b_enc) if sel is None else (self.W_enc[:, sel], self.b_enc[sel])
        return (x - self.b_dec) @ W + b

    def encode(self, x, sel=None):
        import torch
        a = self.pre(x, sel); t = self.thr if sel is None else self.thr[sel]
        return torch.relu(a) * (a > t)

    def decode(self, z):
        return (z @ self.W_dec + self.b_dec) * self.c
