"""Pre-registered analysis for notes/sidefx_confirm.md (clampp5 vs p50chook, paired by problem).
    python scripts/sidefx_confirm_analyze.py <out.json> ref=<glob> trt=<glob> [--fresh-from 200]
Each glob may match several chunk files (merged). Per problem: per-arm mean over its samples. Primary: one-sided sign-flip
permutation test on sum_i (ws_ref_i - ws_trt_i), 100k flips, seed 0. NI: problem-level bootstrap (10k, seed 0) of trt - ref,
one-sided 95% bounds (loop_end upper <= +0.03, acc_all lower >= -0.03). Also completion, acc_on_completed, no-box, cut-after-think,
paired acc, 95% CIs; scopes all / gsm8k / math500 / fresh (problem index >= fresh-from)."""
import sys, json, glob, numpy as np
REPO = __file__.rsplit("/scripts/", 1)[0]
outp = sys.argv[1]; args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a and not a.startswith("--"))
FRESH = 200
idx = {("math500", json.loads(l)["id"]): i for i, l in enumerate(open(f"{REPO}/data/math500.jsonl"))}
idx.update({("gsm8k", json.loads(l)["id"]): i for i, l in enumerate(open(f"{REPO}/data/gsm8k_test.jsonl"))})

def load(pat):
    fs = sorted(glob.glob(pat)); assert fs, pat
    G = {}
    for f in fs:
        for l in open(f):
            x = json.loads(l); x.pop("prompt_ids", None); x.pop("token_ids", None); x.pop("text", None)
            G.setdefault((x["set"], x["id"]), []).append(x)
    return G, fs

ref, fref = load(args["ref"]); trt, ftrt = load(args["trt"])
F = {"whitespace": lambda x: x["cls"] == "cut_whitespace", "loop_end": lambda x: x["loop_end"], "acc_all": lambda x: x["correct"],
     "completion": lambda x: not x["cut"], "cut_after_think": lambda x: x["cut"] and x["closed"],
     "nobox_any": lambda x: (not x["cut"]) and x["pred"] is None}

def permtest(d, n=100_000, seed=0):
    rng = np.random.default_rng(seed); obs = d.sum()
    hits = 0
    for _ in range(n // 10_000):                         # chunks keep memory small
        hits += int(np.sum(rng.choice([-1.0, 1.0], size=(10_000, len(d))) @ d >= obs - 1e-12))
    return float((hits + 1) / (n + 1))

def scope_keys(scope):
    ks = sorted(set(ref) & set(trt))
    if scope in ("gsm8k", "math500"): ks = [k for k in ks if k[0] == scope]
    if scope == "fresh": ks = [k for k in ks if idx[k] >= FRESH]
    return ks

def arm_stats(G, ks):
    xs = [x for k in ks for x in G[k]]; fin = [x for x in xs if not x["cut"]]
    return dict(n_problems=len(ks), n=len(xs), completion=len(fin) / len(xs), acc_on_completed=sum(x["correct"] for x in fin) / max(1, len(fin)),
                acc_all=sum(x["correct"] for x in xs) / len(xs), loop_end=sum(x["loop_end"] for x in xs) / len(xs),
                whitespace=sum(F["whitespace"](x) for x in xs) / len(xs), n_whitespace=sum(F["whitespace"](x) for x in xs),
                nobox_per_finished=sum(x["pred"] is None for x in fin) / max(1, len(fin)), n_nobox=sum(x["pred"] is None for x in fin),
                n_finished=len(fin), cut_after_think=sum(F["cut_after_think"](x) for x in xs) / len(xs),
                mean_tokens=float(np.mean([x["n_tokens"] for x in xs])))

out = dict(files=dict(ref=fref, trt=ftrt))
rng = np.random.default_rng(0)
for scope in ["all", "gsm8k", "math500", "fresh"]:
    ks = scope_keys(scope)
    if not ks: continue
    M = {f: (np.array([np.mean([fn(x) for x in ref[k]]) for k in ks]), np.array([np.mean([fn(x) for x in trt[k]]) for k in ks])) for f, fn in F.items()}
    o = dict(ref=arm_stats(ref, ks), trt=arm_stats(trt, ks))
    B = rng.integers(0, len(ks), size=(10_000, len(ks)))
    for f, (a, b) in M.items():
        d = b - a; bs = d[B].mean(1)
        o[f"diff_{f}"] = dict(trt_minus_ref=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              p05=float(np.percentile(bs, 5)), p95=float(np.percentile(bs, 95)))
    # paired accuracy: problems where both arms finish >= 1 sample (accuracy among finished samples)
    pk = [k for k in ks if any(not x["cut"] for x in ref[k]) and any(not x["cut"] for x in trt[k])]
    pa = lambda G: np.array([np.mean([x["correct"] for x in G[k] if not x["cut"]]) for k in pk])
    ar, at = pa(ref), pa(trt); Bp = rng.integers(0, len(pk), size=(10_000, len(pk)))
    o["paired_acc"] = dict(n=len(pk), ref=float(ar.mean()), trt=float(at.mean()),
                           ci95_diff=[float(np.percentile((at - ar)[Bp].mean(1), q)) for q in (2.5, 97.5)])
    a, b = M["whitespace"]; o["primary_perm_p_one_sided"] = permtest(a - b)
    o["ws_ratio"] = float(b.sum() / a.sum()) if a.sum() else None
    out[scope] = o
A = out["all"]
out["criteria"] = dict(primary_confirmed=A["primary_perm_p_one_sided"] < 0.05, NI_loop=A["diff_loop_end"]["p95"] <= 0.03,
                       NI_acc=A["diff_acc_all"]["p05"] >= -0.03)
json.dump(out, open(outp, "w"), indent=1)
for s in ["all", "gsm8k", "math500", "fresh"]:
    if s not in out: continue
    o = out[s]; print(f"\n## {s}  (problems {o['ref']['n_problems']}, samples/arm {o['ref']['n']})")
    for arm in ("ref", "trt"):
        m = o[arm]
        print(f"{arm}: compl {m['completion']:.3f} x acc_on_compl {m['acc_on_completed']:.3f} = acc {m['acc_all']:.3f} | loop {m['loop_end']:.3f} | "
              f"ws {m['whitespace']:.4f} ({m['n_whitespace']}) | nobox/fin {m['n_nobox']}/{m['n_finished']} | cut-after-think {m['cut_after_think']:.3f} | tok {m['mean_tokens']:.0f}")
    for f in ("whitespace", "loop_end", "acc_all", "completion", "nobox_any", "cut_after_think"):
        d = o[f"diff_{f}"]; print(f"  trt-ref {f}: {d['trt_minus_ref']:+.4f} CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] (5%/95%: {d['p05']:+.4f}/{d['p95']:+.4f})")
    p = o["paired_acc"]; print(f"  paired acc (n={p['n']}): ref {p['ref']:.3f} trt {p['trt']:.3f} diff CI95 [{p['ci95_diff'][0]:+.3f}, {p['ci95_diff'][1]:+.3f}]")
    print(f"  primary one-sided perm p = {o['primary_perm_p_one_sided']:.4g}; ws ratio trt/ref = {o['ws_ratio']}")
print("\nCRITERIA", out["criteria"])
