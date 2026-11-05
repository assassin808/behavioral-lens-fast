"""Arms table for the side-effect experiment.
    python scripts/sidefx_analyze.py <dir with <prefix><arm>.jsonl> <prefix> <out.json> [arms...]
Per arm (sets pooled and per set): completion, loop_end, acc_all, SE1 no-box/finished, SE2 whitespace-cut/all, acc on completed,
paired accuracy vs p50c (problems where both finish >= 1 sample), and problem-level bootstrap 95% CIs of arm - p50c."""
import sys, os, json, numpy as np
REF = os.environ.get("SFX_REF", "p50chook")   # comparator; hook arms are compared with the hook-path p50c
D, pre, outp = sys.argv[1], sys.argv[2], sys.argv[3]
arms = sys.argv[4:] or ["base", "p50c", "p50chook", "clampp5", "loopsel", "clampmean", "a3"]
R = {}
import glob as _g
K0 = os.environ.get("SFX_K0") == "1"          # keep only sample k=0 (arms run with different numbers of samples)
for a in arms:
    if "=" in a: a, f = a.split("=", 1)
    else:
        fs = sorted(_g.glob(f"{D}/{pre}{a}.jsonl")) or sorted(_g.glob(f"{D}/{pre}{a}_n*.jsonl"))
        if not fs: continue
        f = fs[0]
    R[a] = [x for x in (json.loads(l) for l in open(f)) if not (K0 and x["k"] != 0)]
    print(a, f, len(R[a]))

def by_prob(recs):
    g = {}
    for x in recs: g.setdefault((x["set"], x["id"]), []).append(x)
    return g

def metrics(G, keys):
    xs = [x for k in keys for x in G.get(k, [])]
    n = len(xs); fin = [x for x in xs if not x["cut"]]
    return dict(n=n, completion=len(fin) / n, loop_end=sum(x["loop_end"] for x in xs) / n, acc_all=sum(x["correct"] for x in xs) / n,
                se1_nobox_per_finished=sum(x["pred"] is None for x in fin) / max(1, len(fin)),
                se2_whitespace=sum(x["cls"] == "cut_whitespace" for x in xs) / n,
                xse_cut_after_think=sum(x["cut"] and x["closed"] for x in xs) / n,       # exploratory (added after the pilot)
                xse_noanswer_any=sum(x["pred"] is None for x in xs) / n,
                acc_on_completed=sum(x["correct"] for x in fin) / max(1, len(fin)),
                n_finished=len(fin), n_nobox=sum(x["pred"] is None for x in fin), n_whitespace=sum(x["cls"] == "cut_whitespace" for x in xs))

def paired(Ga, Gb, keys):
    ks = [k for k in keys if any(not x["cut"] for x in Ga.get(k, [])) and any(not x["cut"] for x in Gb.get(k, []))]
    acc = lambda G: np.mean([np.mean([x["correct"] for x in G[k] if not x["cut"]]) for k in ks]) if ks else float("nan")
    return len(ks), acc(Ga), acc(Gb)

out = {}
G = {a: by_prob(r) for a, r in R.items()}
rng = np.random.default_rng(0)
for scope in ["all", "gsm8k", "math500"]:
    out[scope] = {}
    for a in R:
        keys = sorted(k for k in G[a] if scope == "all" or k[0] == scope)
        m = metrics(G[a], keys)
        if REF in G and a != REF:
            npair, pa, pp = paired(G[a], G[REF], keys)
            m.update(paired_n=npair, paired_acc=pa, paired_acc_ref=pp)
            common = [k for k in keys if k in G[REF]]
            B = []
            for _ in range(1000):
                s = [common[i] for i in rng.integers(0, len(common), len(common))]
                ma, mp = metrics(G[a], s), metrics(G[REF], s)
                B.append([ma[f] - mp[f] for f in ("completion", "loop_end", "se1_nobox_per_finished", "se2_whitespace", "xse_cut_after_think", "acc_all")])
            B = np.array(B)
            m["ci95_minus_ref"] = {f: [float(np.percentile(B[:, i], 2.5)), float(np.percentile(B[:, i], 97.5))]
                                    for i, f in enumerate(("completion", "loop_end", "se1_nobox_per_finished", "se2_whitespace", "xse_cut_after_think", "acc_all"))}
        cls = {}
        for k in keys:
            for x in G[a][k]: cls[x["cls"]] = cls.get(x["cls"], 0) + 1
        m["classes"] = cls
        out[scope][a] = m
json.dump(out, open(outp, "w"), indent=1)
for scope in out:
    print(f"\n## {scope}")
    print("arm | n | compl | × acc_on_compl = acc_all | loop_end | no-box/fin | whitespace | cut after </think> | acc_on_compl | paired acc (arm/ref, n)")
    for a, m in out[scope].items():
        pa = f"{m['paired_acc']:.3f}/{m['paired_acc_ref']:.3f} ({m['paired_n']})" if "paired_n" in m else "-"
        print(f"{a} | {m['n']} | {m['completion']:.3f} × {m['acc_on_completed']:.3f} = {m['acc_all']:.3f} | {m['loop_end']:.3f} | {m['se1_nobox_per_finished']:.3f} ({m['n_nobox']}/{m['n_finished']}) | "
              f"{m['se2_whitespace']:.3f} ({m['n_whitespace']}) | {m['xse_cut_after_think']:.3f} | {m['acc_on_completed']:.3f} | {pa}")
