"""Analysis of the LLM failure labels (notes/api_failure_labels.md).
    python scripts/api/fail_analyze.py <out.json>
Reads notes/api_fail_labels_claude-sonnet-5-5_all.json (+ kappa50 files of Sonnet and gpt-oss-120b). Re-parses malformed JSON with the
field-wise fallback in fail_label.parse. Correct-stratum items of base_nothink / p50c_nothink are weighted by population/sample size
where pooled over strata (Q3)."""
import json, sys, collections, random, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fail_label as F
OUT = sys.argv[1]; assert not os.path.exists(OUT), "exists; not overwriting"
R = json.load(open(f"{F.REPO}/notes/api_fail_labels_claude-sonnet-5-5_all.json"))["items"]
for x in R: x["L"] = x["label"] or F.parse(x["raw"])
nfail = sum(x["L"] is None for x in R); R = [x for x in R if x["L"] is not None]
ARMS = ["p50c_think", "base_nothink", "p50c_nothink", "base_think", "p50chook_think", "clampp5_think", "loopsel_think", "clampmean_think", "a3_think"]
TYPES = [c for c in F.CODES if c != "correct"]
res = dict(n_items=len(R) + nfail, unparseable=nfail, fieldwise=sum(bool(x["L"].get("_fieldwise")) for x in R))
def boot(fun, xs, B=2000, seed=0):
    rng = random.Random(seed); v = []
    for _ in range(B): v.append(fun([xs[rng.randrange(len(xs))] for _ in xs]))
    v = [a for a in v if a is not None and not np.isnan(a)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
# ---- T1: failure-type shares per arm (failures only) ----
T1 = {}
for a in ARMS:
    f = [x for x in R if x["arm"] == a and not x["correct"]]
    if not f: continue
    c = collections.Counter(x["L"]["main_type"] for x in f)
    T1[a] = dict(n_fail=len(f), share={t: c[t] / len(f) for t in F.CODES}, n_correct_label_on_fail=c["correct"],
                 wording_doubt_present=float(np.mean([bool(x["L"]["wording_doubt_present"]) for x in f])),
                 final_answer_reached=float(np.mean([bool(x["L"]["final_answer_reached"]) for x in f])),
                 any_cycling=float(np.mean([x["L"]["main_type"] == "paraphrase_cycling" or "paraphrase_cycling" in x["L"]["also_present"] for x in f])),
                 any_over_doubt=float(np.mean([x["L"]["main_type"] == "over_doubt_wording" or "over_doubt_wording" in x["L"]["also_present"] for x in f])),
                 cut_rate=float(np.mean([x["cut"] for x in f])))
    for sub, sel in [("finished", lambda x: not x["cut"]), ("cut", lambda x: x["cut"])]:
        g = [x for x in f if sel(x)]; cc = collections.Counter(x["L"]["main_type"] for x in g)
        T1[a][f"n_{sub}"] = len(g); T1[a][f"share_{sub}"] = {t: cc[t] / len(g) for t in F.CODES} if g else None
res["T1_failure_types"] = T1
# ---- F1 / Q1: over_doubt_wording share, p50c think vs base nothink failures ----
def share(code):
    return lambda xs: float(np.mean([x["L"]["main_type"] == code for x in xs])) if xs else None
pf = [x for x in R if x["arm"] == "p50c_think" and not x["correct"]]; bf = [x for x in R if x["arm"] == "base_nothink" and not x["correct"]]
def diffci(fun):
    d = fun(pf) - fun(bf); rng = random.Random(1); v = []
    for _ in range(2000):
        v.append(fun([pf[rng.randrange(len(pf))] for _ in pf]) - fun([bf[rng.randrange(len(bf))] for _ in bf]))
    return dict(p50c=fun(pf), base_nothink=fun(bf), diff=d, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
wd = lambda xs: float(np.mean([bool(x["L"]["wording_doubt_present"]) for x in xs]))
anyod = lambda xs: float(np.mean([x["L"]["main_type"] == "over_doubt_wording" or "over_doubt_wording" in x["L"]["also_present"] for x in xs]))
res["F1"] = dict(main_over_doubt=diffci(share("over_doubt_wording")), wording_doubt_present=diffci(wd), any_over_doubt=diffci(anyod))
# same restricted to finished-wrong outputs (wrong-but-completed) on both sides
pf0, bf0 = pf, bf
pf = [x for x in pf0 if not x["cut"]]; bf = [x for x in bf0 if not x["cut"]]
res["F1_finished_wrong_only"] = dict(n=[len(pf), len(bf)], main_over_doubt=diffci(share("over_doubt_wording")), wording_doubt_present=diffci(wd))
pf = [x for x in pf0 if x["set"] == "gsm8k"]; bf = [x for x in bf0 if x["set"] == "gsm8k"]
res["F1_gsm8k_only"] = dict(n=[len(pf), len(bf)], main_over_doubt=diffci(share("over_doubt_wording")), wording_doubt_present=diffci(wd))
pf = [x for x in pf0 if x["set"] == "math500"]; bf = [x for x in bf0 if x["set"] == "math500"]
res["F1_math_only"] = dict(n=[len(pf), len(bf)], main_over_doubt=diffci(share("over_doubt_wording")), wording_doubt_present=diffci(wd))
pf, bf = pf0, bf0
# ---- F2 / Q2: cycling in long unfinished p50c outputs ----
F2 = {}
for a in ARMS:
    g = [x for x in R if x["arm"] == a and x["cut"] and x["L"]["main_type"] not in ("verbatim_loop", "degenerate")]
    if g:
        cyc = lambda xs: float(np.mean([x["L"]["main_type"] == "paraphrase_cycling" or "paraphrase_cycling" in x["L"]["also_present"] for x in xs]))
        F2[a] = dict(n=len(g), cycling_main=share("paraphrase_cycling")(g), cycling_main_or_also=cyc(g), ci=boot(cyc, g),
                     main_types=dict(collections.Counter(x["L"]["main_type"] for x in g)))
res["F2_cycling_in_cut_nonloop"] = F2
# ---- F3 / Q3: doubt after first stated answer vs its correctness (weighted for stratified arms) ----
def w(x):
    if x["stratum"] == "correct": return x["n_corr_pop"] / sum(1 for y in R if y["arm"] == x["arm"] and y["stratum"] == "correct")
    if x["stratum"] == "fail" and x["arm"] == "p50c_nothink": return x["n_fail_pop"] / sum(1 for y in R if y["arm"] == x["arm"] and y["stratum"] == "fail")
    return 1.0
def pdoubt(xs, st):
    g = [x for x in xs if x["L"]["first_answer_status"] == st and x["L"]["doubt_after_first"] in ("yes", "no")]
    if not g: return None
    return float(sum(w(x) * (x["L"]["doubt_after_first"] == "yes") for x in g) / sum(w(x) for x in g))
F3 = {}
for a in ARMS:
    xs = [x for x in R if x["arm"] == a]
    if not xs: continue
    st = collections.Counter(x["L"]["first_answer_status"] for x in xs)
    pc, pw = pdoubt(xs, "correct"), pdoubt(xs, "wrong")
    d = (pc - pw) if pc is not None and pw is not None else None
    F3[a] = dict(n=len(xs), status=dict(st), p_doubt_given_first_correct=pc, p_doubt_given_first_wrong=pw, diff=d,
                 ci=boot(lambda ys: (pdoubt(ys, "correct") - pdoubt(ys, "wrong")) if pdoubt(ys, "correct") is not None and pdoubt(ys, "wrong") is not None else None, xs) if d is not None else None,
                 n_correct_first=sum(1 for x in xs if x["L"]["first_answer_status"] == "correct" and x["L"]["doubt_after_first"] in ("yes", "no")),
                 n_wrong_first=sum(1 for x in xs if x["L"]["first_answer_status"] == "wrong" and x["L"]["doubt_after_first"] in ("yes", "no")),
                 doubt_target_given_doubt=dict(collections.Counter(x["L"]["doubt_target"] for x in xs if x["L"]["doubt_after_first"] == "yes")),
                 # outcome after a correct first answer: share that end correct
                 p_final_correct_given_first_correct=float(np.mean([x["correct"] for x in xs if x["L"]["first_answer_status"] == "correct"])) if st["correct"] else None)
res["F3_doubt_after_first"] = F3
# ---- validation: kappa Sonnet vs gpt-oss on the kappa50 set ----
def kappa(a, b):
    cats = sorted(set(a) | set(b), key=str); n = len(a); po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats); return float((po - pe) / (1 - pe)) if pe < 1 else 1.0, float(po)
S = {x["uid"]: x for x in json.load(open(f"{F.REPO}/notes/api_fail_labels_claude-sonnet-5-5_kappa50.json"))["items"]}
O = {x["uid"]: x for x in json.load(open(f"{F.REPO}/notes/api_fail_labels_gpt-oss-120b_kappa50.json"))["items"]}
pairs = []
for u in S:
    ls, lo = S[u]["label"] or F.parse(S[u]["raw"]), O[u]["label"] or F.parse(O[u]["raw"])
    if ls and lo: pairs.append((S[u], ls, lo))
K = dict(n_pairs=len(pairs))
for fld in ["main_type", "wording_doubt_present", "doubt_after_first", "first_answer_status", "final_answer_reached"]:
    for sub, sel in [("all", lambda it: True), ("failures", lambda it: not it["correct"])]:
        g = [(ls[fld], lo[fld]) for it, ls, lo in pairs if sel(it)]
        k, po = kappa([str(a) for a, _ in g], [str(b) for _, b in g]); K[f"{fld}_{sub}"] = dict(n=len(g), kappa=k, agree=po)
# coarse 4-way: doubt-type (over_doubt/endless/cycling) vs loop/degenerate vs error vs correct/other
def coarse(t): return {"over_doubt_wording": "doubt", "endless_recheck": "doubt", "paraphrase_cycling": "doubt", "verbatim_loop": "loop",
                       "degenerate": "loop", "reasoning_error": "error"}.get(t, t)
g = [(coarse(ls["main_type"]), coarse(lo["main_type"])) for it, ls, lo in pairs if not it["correct"]]
k, po = kappa([a for a, _ in g], [b for _, b in g]); K["main_type_coarse_failures"] = dict(n=len(g), kappa=k, agree=po)
K["confusion_main_type_failures"] = dict(collections.Counter(f"{ls['main_type']} | {lo['main_type']}" for it, ls, lo in pairs if not it["correct"]))
res["kappa_sonnet_vs_gptoss"] = K
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1)[:12000])
