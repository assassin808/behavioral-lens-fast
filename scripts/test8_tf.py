"""Test 8 (GPU part): loop probe vs u, and the probe run over generations of several (model, text) pairs.

1. Fit a logistic loop probe per layer on BASE states (from scripts/test3_collect.py: up to 256 loop + 256 non-loop positions per
   math context, thinking-on texts, 16k), standardized space, C=0.1 (same recipe as scripts/q5_state_geometry.py); 5-fold
   context-grouped CV accuracy; map the final probe back to the raw residual space: w_raw = w_std / sd, b_raw = b - sum(w_std*mu/sd).
2. Teacher-force passes; per position: resid.u15 at hidden_states[15/16], resid.u19 at [19/20/24], probe logit at each probe layer,
   32-gram and 16-gram loop masks; sampled states at hidden_states[16], [20] for the PCA figure.
   passes: p50c on p50c's own thinking-on generations (GSM8K+MATH, Astra logs)
           base on the same p50c generations (how the base model reads them)
           base on base thinking-on loop texts; p50c on the same texts (edited states at base loop positions)
           base on base thinking-OFF generations (scripts/test8_gen_nothink.py): all with loop_frac > 0.02 + 20 clean
           base on first 2048 generated tokens of thinking-on vs thinking-off generations of the same 400 problems
    python scripts/test8_tf.py [MAXT] [N]          out: $T38_DATA/t8_tf_{MAXT}.pt
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODEL_DIR, REPO, DATA_OUT, astra_rows, encode, loop_mask
DEV = os.environ.get("BLF_DEVICE", "cuda")
from sklearn.linear_model import LogisticRegression

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
PILOT = bool(os.environ.get("PILOT"))
PL = [15, 16, 19, 20, 24]
D = torch.load(f"{DATA_OUT}/t3_collect_{MAXT}{'_pilot' if PILOT and not os.path.exists(f'{DATA_OUT}/t3_collect_{MAXT}.pt') else ''}.pt",
               weights_only=False)
U = {L: np.load(f"{REPO}/shared/u_L{L}.npy").astype(np.float64) for L in [15, 19]}
uL = {15: U[15], 16: U[15], 19: U[19], 20: U[19], 24: U[19]}

# ---------------- 1. probe
st = D["states"]; groups = np.concatenate([np.full(len(s["pos"]), s["ctx"]) for s in st]); y = np.concatenate([s["is_loop"] for s in st]).astype(int)
SUB = np.sort(np.random.default_rng(0).choice(len(y), min(len(y), 40000), replace=False))   # cap probe-fit cost
groups, y = groups[SUB], y[SUB]
probe, report = {}, {}
for L in PL:
    X = torch.cat([s["H"][L] for s in st])[torch.from_numpy(SUB)].float().numpy()
    ug = np.unique(groups); rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(ug), 5)
    accs, aucs_u = [], []
    from sklearn.metrics import roc_auc_score
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if y[te].min() == y[te].max(): continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        accs.append(0.5 * ((p[y[te] == 1] > .5).mean() + (p[y[te] == 0] <= .5).mean()))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit((X - mu) / sd, y)
    w_std = clf.coef_[0]; w_raw = w_std / sd; b_raw = clf.intercept_[0] - (w_std * mu / sd).sum()
    dm = X[y == 1].mean(0) - X[y == 0].mean(0)
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    proj_u = X @ uL[L]
    report[L] = dict(n_states=int(len(y)), n_loop=int(y.sum()), n_ctx=int(len(ug)), cv_bal_acc=float(np.mean(accs)),
                     cos_probe_raw_u=cos(w_raw, uL[L]), cos_probe_std_u=cos(w_std, uL[L]), cos_diffmeans_u=cos(dm, uL[L]),
                     cos_probe_raw_u15=cos(w_raw, U[15]), cos_probe_raw_u19=cos(w_raw, U[19]),
                     cos_probe_raw_diffmeans=cos(w_raw, dm), auc_resid_u_alone=float(roc_auc_score(y, proj_u)),
                     auc_probe_insample=float(roc_auc_score(y, X @ w_raw)),
                     frac_probe_logit_var_from_u=float(np.corrcoef(X @ w_raw, (proj_u[:, None] * uL[L][None]) @ w_raw)[0, 1] ** 2))
    probe[L] = (torch.tensor(w_raw, dtype=torch.float32), float(b_raw))
    print("probe", L, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in report[L].items()}), flush=True)

# ---------------- 2. passes
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_DIR["base"])
def texts_p50c():
    R = [r for ds in ["gsm8k", "math500"] for r in astra_rows("p50c", ds)]
    rng = np.random.default_rng(1); R = [R[i] for i in rng.permutation(len(R))[:N]]
    return [dict(id=r["id"], src=r["src"], text=r["text"], think=True, closed="</think>" in r["text"]) for r in R]
def texts_base_loop():
    loopm = [m for m in D["meta"] if m["cls"] == "loop"][: N // 2]
    byid = {r["id"]: r for ds in ["gsm8k", "math500"] for r in astra_rows("unedited", ds)}
    return [dict(id=m["id"], src=m["src"], text=byid[m["id"]]["text"], think=True) for m in loopm]
G = [json.loads(l) for l in open(f"{DATA_OUT}/gen_base_nothink.jsonl")] if os.path.exists(f"{DATA_OUT}/gen_base_nothink.jsonl") else []
def texts_nothink():
    lo = [g for g in G if g["loop_frac"] > 0.02]; cl = [g for g in G if g["loop_frac"] == 0 and not g["cut"]][:20]
    return [dict(id=g["id"], src=g["src"], text=g["text"], think=False, nt_loop=g["loop_frac"] > 0.02) for g in lo[:N] + cl]
def texts_first(think):
    if think:
        R = [r for ds in ["gsm8k", "math500"] for r in astra_rows("unedited", ds)]
        return [dict(id=r["id"], src=r["src"], text=r["text"], think=True) for r in R]
    return [dict(id=g["id"], src=g["src"], text=g["text"], think=False) for g in G]

PASSES = [("p50c", "p50c_own", texts_p50c, MAXT), ("base", "p50c_own", texts_p50c, MAXT),
          ("base", "base_loop", texts_base_loop, MAXT), ("p50c", "base_loop", texts_base_loop, MAXT),
          ("base", "base_nothink", texts_nothink, MAXT),
          ("base", "first_think", lambda: texts_first(True), 2048 + 300), ("base", "first_nothink", lambda: texts_first(False), 2048 + 300)]
out = dict(probe_report=report, probe={L: (w.numpy(), b) for L, (w, b) in probe.items()}, passes={})
for tag in ["base", "p50c"]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR[tag], dtype=torch.bfloat16).to(DEV).eval()
    Uc = {L: torch.tensor(uL[L], dtype=torch.float32, device=DEV) for L in PL}
    Wc = {L: probe[L][0].to(DEV) for L in PL}
    for (mtag, name, fn, maxt) in PASSES:
        if mtag != tag: continue
        T0 = fn()
        if PILOT: T0 = T0[:6]
        if os.environ.get("SMOKE_N"): T0 = T0[:int(os.environ["SMOKE_N"])]
        if not T0: print(f"pass {mtag}|{name}: no texts, skipped", flush=True); continue
        recs = []; t0 = time.time()
        with torch.no_grad():
            for i, t in enumerate(T0):
                ids, P = encode(tok, t["id"], t["text"], think=t["think"], maxt=maxt)
                hs = model.model(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
                r = dict(id=t["id"], src=t["src"], P=P, T=len(ids), m32=loop_mask(ids, 32), m16=loop_mask(ids, 16),
                         **{k: v for k, v in t.items() if k not in ("text", "id", "src")})
                for L in PL:
                    h = hs[L][0].float()
                    r[f"u_{L}"] = (h @ Uc[L]).half().cpu().numpy()
                    r[f"logit_{L}"] = (h @ Wc[L] + probe[L][1]).half().cpu().numpy()
                if name in ("base_loop", "p50c_own"):
                    R = np.random.default_rng(i); idx = np.sort(R.choice(np.arange(P, len(ids)), min(256, len(ids) - P), replace=False))
                    r["pca_idx"] = idx; r["pca_H16"] = hs[16][0][idx].half().cpu().numpy(); r["pca_H20"] = hs[20][0][idx].half().cpu().numpy()
                recs.append(r); del hs
        out["passes"][f"{mtag}|{name}"] = recs
        lf = np.mean([r["m32"][r["P"]:].mean() for r in recs])
        print(f"pass {mtag}|{name}: {len(recs)} texts, mean T {np.mean([r['T'] for r in recs]):.0f}, loop_frac32 {lf:.3f}, {time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
outp = f"{DATA_OUT}/t8_tf_{MAXT}{'_pilot' if PILOT else ''}.pt"
torch.save(out, outp); json.dump(report, open(f"{DATA_OUT}/t8_probe_report_{MAXT}{'_pilot' if PILOT else ''}.json", "w"), indent=1)
print("saved", outp, flush=True)
