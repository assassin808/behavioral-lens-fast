"""Experiment B: paraphrase-loop detector (pre-registered in notes/expB_paraphrase_loops.md).

Stages:
  build   -> collect texts of every group, classify, segment, embed, score pairs; writes {out}/texts_{tag}.jsonl.gz + emb npz
  calib   -> draw 60 stratified non-verbatim pairs (group hidden) for hand labelling -> {out}/calib_pairs.json
  report  -> PR metrics per group/class at given tau -> {out}/report_tau{tau}.json
All groups go through the same segmentation / embedding / verbatim test (shared harness).
"""
import argparse, glob, gzip, json, os, re, sys, random
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
EMB = "/scratch/yang0531/blf_data/expB/emb/all-MiniLM-L6-v2"
TOK = "/scratch/yang0531/blf_models/base"
NOTHINK = "/scratch/yang0531/blf_data/nothink_runpod"
LOOKBACK = (2, 11)  # j in [i-11, i-2]


# ---------------------------------------------------------------- data
def classify(text, answer):
    # identical to scripts/failure_classes_from_logs.py
    from astra_eval import grade, loop_signals
    ok, pred = grade(text, answer)
    if ok: return "correct"
    closed = "</think>" in text
    tail = text[-2000:]
    if not closed:
        ls = loop_signals(text)
        if re.fullmatch(r"\s*", tail): return "cut: degenerate whitespace"
        if ls["verbatim_loop_end"]: return "cut: verbatim loop"
        if ls["repetitive_end"]: return "cut: repetitive (paraphrase)"
        return "cut: still reasoning"
    if pred is None: return "finished: no \\boxed{}"
    return "finished: wrong answer"


def preloop(text):
    """truncate at first word position of any 32-word n-gram occurring >= 3 times."""
    w = text.split()
    cnt, first = {}, {}
    for k in range(len(w) - 31):
        g = " ".join(w[k:k + 32]); cnt[g] = cnt.get(g, 0) + 1
        first.setdefault(g, k)
    starts = [first[g] for g, c in cnt.items() if c >= 3]
    if not starts: return text, False
    k0 = min(starts)
    # map word index back to char offset
    pos = 0
    for m, mt in enumerate(re.finditer(r"\S+", text)):
        if m == k0: pos = mt.start(); break
    return text[:pos], True


def _cls(x): return classify(*x)


def load_groups(sets, procs=16):
    from multiprocessing import Pool
    G = []; raw = []
    for run, grp in [("p50c", "p50c_think"), ("unedited", "base_think")]:
        for ds in sets:
            for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/{run}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
                for l in gzip.open(f, "rt"): raw.append((grp, ds, json.loads(l)))
    with Pool(procs) as pool: C = pool.map(_cls, [(r["text"], r["answer"]) for _, _, r in raw], chunksize=4)
    if True:
        for (grp, ds, r), cls in zip(raw, C):
                if True:
                    t = r["text"]; th = t.split("</think>")[0]
                    if grp == "base_think":
                        pre, had = preloop(th)
                        G.append(dict(group="base_think_preloop", ds=ds, id=r["id"], cls=cls, text=pre, had_loop=had))
                    else:
                        G.append(dict(group=grp, ds=ds, id=r["id"], cls=cls, text=th))
    for tag in ["base", "p50c"]:
        for l in open(f"{NOTHINK}/{tag}_nothink_runpod.jsonl"):
            r = json.loads(l)
            if r["set"] not in sets: continue
            cls = "correct" if r["correct"] else ("cut" if r["cut"] else "finished: wrong/no box")
            G.append(dict(group=f"{tag}_nothink", ds=r["set"], id=f'{r["id"]}#{r["k"]}', cls=cls, text=r["text"]))
    return G


# ---------------------------------------------------------------- segmentation
def segment(text):
    paras = [p.split() for p in re.split(r"\n\s*\n", text) if p.strip()]
    segs, cur = [], []
    for p in paras:
        cur += p
        if len(cur) >= 40: segs.append(cur); cur = []
    if cur:
        if segs: segs[-1] += cur
        else: segs.append(cur)
    out = []
    for s in segs:
        if len(s) > 160:
            ch = [s[k:k + 80] for k in range(0, len(s), 80)]
            if len(ch) > 1 and len(ch[-1]) < 40: ch[-2] += ch[-1]; ch = ch[:-1]
            out += ch
        else:
            out.append(s)
    return [" ".join(s) for s in out]


LEX = re.compile(r"\\[a-zA-Z]+|[a-z]+|\d+(?:\.\d+)?")
def lexset(s): return set(LEX.findall(s.lower()))
def ngrams(seq, n): return {tuple(seq[k:k + n]) for k in range(len(seq) - n + 1)}


# ---------------------------------------------------------------- build
_W = {}
def _init(threads):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(threads)
    _W["q"] = AutoTokenizer.from_pretrained(TOK)
    _W["et"] = AutoTokenizer.from_pretrained(EMB); _W["em"] = AutoModel.from_pretrained(EMB).eval()


def _embed(xs):
    import torch
    etok, emod = _W["et"], _W["em"]; outs = []
    for k in range(0, len(xs), 256):
        b = etok(xs[k:k + 256], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad(): h = emod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float(); e = (h * m).sum(1) / m.sum(1)
        outs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 384), np.float32)


def _vpairs(grams):
    idx = {}
    for k, gs in enumerate(grams):
        for x in gs: idx.setdefault(x, set()).add(k)
    vp = set()
    for ks in {frozenset(v) for v in idx.values() if len(v) > 1}:
        ks = sorted(ks); vp |= {(ks[b], ks[c]) for c in range(len(ks)) for b in range(c)}
    return vp  # (j, i) with j < i


def process(g):
    segs = segment(g["text"])
    E = _embed(segs)
    qt = [_W["q"](s, add_special_tokens=False)["input_ids"] for s in segs]
    V32 = _vpairs([ngrams(t, 32) for t in qt]); V12 = _vpairs([ngrams(s.split(), 12) for s in segs])
    lx = [lexset(s) for s in segs]; voc = {w: k for k, w in enumerate(set().union(*lx))} if lx else {}
    X = np.zeros((len(segs), max(1, len(voc))), np.float32)
    for k, ws in enumerate(lx):
        for w in ws: X[k, voc[w]] = 1
    inter = X @ X.T; sz = X.sum(1); J = inter / np.maximum(1, sz[:, None] + sz[None, :] - inter)
    C = E @ E.T if len(segs) else np.zeros((0, 0))
    pairs = []
    for i in range(len(segs)):
        for j in range(0, i - 1):
            v32 = (j, i) in V32
            if C[i, j] >= 0.5 or J[i, j] >= 0.5 or v32:
                pairs.append([i, j, round(float(C[i, j]), 4), round(float(J[i, j]), 4), int(v32), int((j, i) in V12)])
    g2 = {k: v for k, v in g.items() if k != "text"}
    g2.update(n_seg=len(segs), n_tok=sum(len(t) for t in qt), n_words=len(g["text"].split()),
              seg_start_word=list(np.cumsum([0] + [len(s.split()) for s in segs[:-1]]).tolist()), pairs=pairs, segs=segs)
    return json.dumps(g2)


def build(a):
    from multiprocessing import Pool
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/texts_{a.tag}.jsonl.gz"
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    G = load_groups(a.sets.split(","), a.procs)
    print("classified", len(G), flush=True)
    if a.limit:
        random.seed(0); byg = {}
        for g in G: byg.setdefault(g["group"], []).append(g)
        G = [x for v in byg.values() for x in random.sample(v, min(a.limit, len(v)))]
    out = gzip.open(path, "wt")
    with Pool(a.procs, initializer=_init, initargs=(1,)) as pool:
        for n, line in enumerate(pool.imap(process, G, chunksize=2)):
            out.write(line + "\n")
            if n % 100 == 0: print(n, len(G), flush=True)
    out.close()


# ---------------------------------------------------------------- metrics
def text_metrics(r, tau, sim="cos", strict=False):
    i_sim = 2 if sim == "cos" else 3
    ns = r["n_seg"]; by_i = {}
    for p in r["pairs"]: by_i.setdefault(p[0], []).append(p)
    def is_pr(p): return p[i_sim] >= tau and not p[4] and not (strict and p[5])
    pr_win = vb_win = elig = pr_all = 0; pr_pos = el_pos = 0
    for i in range(ns):
        ps = by_i.get(i, [])
        win = [p for p in ps if i - 11 <= p[1] <= i - 2]
        if any(is_pr(p) for p in ps): pr_all += 1
        if i >= 11:
            elig += 1; hit = any(is_pr(p) for p in win)
            pr_win += hit; vb_win += any(p[4] for p in win)
            if r["seg_start_word"][i] < 2000: el_pos += 1; pr_pos += hit
    return dict(elig=elig, pr_rate=pr_win / elig if elig else None, vb_rate=vb_win / elig if elig else None,
                pr_per_1k=1000 * pr_all / max(1, r["n_tok"]), pr_rate_pos=pr_pos / el_pos if el_pos else None)


def load_built(out):
    R = []
    for f in sorted(glob.glob(f"{out}/texts_*.jsonl.gz")):
        if "pilot" in f: continue
        R += [json.loads(l) for l in gzip.open(f, "rt")]
    return R


def boot_diff(x, y, B=5000, seed=0):
    rng = np.random.default_rng(seed); x = np.array(x); y = np.array(y)
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(B)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(a):
    R = load_built(a.out)
    res = {}
    for sim, taus in [("cos", [a.tau, round(a.tau + 0.05, 3)]), ("lex", [0.5, 0.6])]:
        for tau in taus:
            for strict in [False, True]:
                key = f"{sim}>={tau}{' strict12' if strict else ''}"
                rows = {}
                for r in R:
                    m = text_metrics(r, tau, sim, strict)
                    cls = r["cls"] if r["group"] in ("p50c_think", "base_think_preloop") else "all"
                    for gk in [(r["group"], cls), (r["group"], "ALL")] if cls != "all" else [(r["group"], "ALL")]:
                        rows.setdefault(f"{gk[0]} | {gk[1]}", []).append(m)
                tab = {}
                for k, ms in rows.items():
                    v = [m for m in ms if m["pr_rate"] is not None]
                    tab[k] = dict(n=len(ms), n_elig=len(v),
                                  pr_rate=float(np.mean([m["pr_rate"] for m in v])) if v else None,
                                  frac_cycling=float(np.mean([m["pr_rate"] >= 0.30 for m in v])) if v else None,
                                  vb_rate=float(np.mean([m["vb_rate"] for m in v])) if v else None,
                                  pr_per_1k=float(np.mean([m["pr_per_1k"] for m in ms])),
                                  pr_rate_pos=float(np.mean([m["pr_rate_pos"] for m in ms if m["pr_rate_pos"] is not None])) if any(m["pr_rate_pos"] is not None for m in ms) else None,
                                  _vals=[m["pr_rate"] for m in v])
                tgt = tab.get("p50c_think | cut: still reasoning")
                tests = {}
                if tgt:
                    for ctl in ["base_nothink | ALL", "p50c_think | correct", "p50c_nothink | ALL", "base_think_preloop | ALL"]:
                        if ctl in tab and tab[ctl]["_vals"]:
                            lo, hi = boot_diff(tgt["_vals"], tab[ctl]["_vals"])
                            tests[ctl] = dict(diff_ci=[lo, hi], ratio=tgt["pr_rate"] / max(1e-9, tab[ctl]["pr_rate"]))
                p3 = None
                if "p50c_think | ALL" in tab and "base_think_preloop | ALL" in tab:
                    p3 = dict(ratio=tab["p50c_think | ALL"]["pr_rate"] / max(1e-9, tab["base_think_preloop | ALL"]["pr_rate"]),
                              diff_ci=boot_diff(tab["p50c_think | ALL"]["_vals"], tab["base_think_preloop | ALL"]["_vals"]))
                for v in tab.values(): v.pop("_vals")
                res[key] = dict(table=tab, P2_tests=tests, P3=p3)
    path = f"{a.out}/report_{a.name}.json"
    json.dump(res, open(path, "w"), indent=1); print("wrote", path)


def calib(a):
    R = load_built(a.out); rng = random.Random(1)
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    pool = {b: [] for b in bins}
    for t, r in enumerate(R):
        for p in r["pairs"]:
            if p[4] or not (p[0] - 11 <= p[1] <= p[0] - 2): continue
            for b in bins:
                if b[0] <= p[2] < b[1]: pool[b].append((t, p))
    out = []
    for b in bins:
        for t, p in rng.sample(pool[b], min(12, len(pool[b]))):
            r = R[t]
            out.append(dict(bin=b[0], cos=p[2], lex=p[3], v12=p[5], group=r["group"], cls=r["cls"], id=r["id"],
                            seg_j=r["segs"][p[1]], seg_i=r["segs"][p[0]], i=p[0], j=p[1]))
    rng.shuffle(out)
    json.dump(out, open(f"{a.out}/calib_pairs.json", "w"), indent=1)
    print({b: len(v) for b, v in pool.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--out", default="/scratch/yang0531/blf_data/expB")
    ap.add_argument("--tag", default="full"); ap.add_argument("--sets", default="math500,gsm8k")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--name", default="main")
    a = ap.parse_args()
    {"build": build, "calib": calib, "report": report}[a.stage](a)
