"""Test 3 (+ test-8 probe training data): teacher-force the BASE model on long real generations and record, for every
down_proj neuron j of L15 / L19, its signed write along u:  c_j(t) = act_j(t) * (u^T w_j)   (base weights, u = shared/u_L*.npy).

Corpus (thinking-on texts = Astra's logged unedited first samples, 32k-cap generations; truncated to MAXT tokens incl. prompt):
  loop    : unedited GSM8K + MATH-500 samples cut at 32k (no </think>) with loop_frac >= 0.05 in the first MAXT tokens
  normal  : unedited GSM8K + MATH-500 samples that finished (</think> present), plus base thinking-OFF generations that finished
            with loop_frac < 0.01 (scripts/test8_gen_nothink.py) to reach >= 50
  nonmath : unedited IFEval + MMLU-Pro samples (all, loop and non-loop positions)
Per context and neuron: sum / sumsq / count of c over loop and non-loop positions (generation tokens only).
Exemplars: for 128 Astra neurons + 16 random neurons per layer, top peaks per context (>= 64 tokens apart, 7-token peak context
de-duplicated) with a 33-token window; plus random windows (half loop, half non-loop positions) with all tracked neurons' acts.
Test-8 data: hidden_states[L] for L in PROBE_LAYERS at up to 256 loop + 256 non-loop positions per math context (float16).

    python scripts/test3_collect.py [MAXT] [N_PER_CLASS]       out: $T38_DATA/t3_collect_{MAXT}.pt
"""
import os, sys, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import MODEL_DIR, RUNTAG, weights_file, REPO, DATA_OUT, astra_rows, encode, loop_mask
DEV = os.environ.get("BLF_DEVICE", "cuda")

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
NPC = int(sys.argv[2]) if len(sys.argv) > 2 else 60
PROBE_LAYERS = [15, 16, 19, 20, 24]
LAYERS = [15, 19]
W_BEFORE, W_AFTER = 24, 8
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_DIR["base"])
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR["base"], dtype=torch.bfloat16).to(DEV).eval()
import transformers.models.qwen3_5.modeling_qwen3_5 as M
print("gated-delta impl:", {k: getattr(getattr(M, k), "__module__", "?") for k in dir(M) if "gated_delta_rule" in k}, flush=True)
layers = model.model.layers

U = {L: torch.tensor(np.load(f"{REPO}/shared/u_L{L}.npy"), dtype=torch.float32, device=DEV) for L in LAYERS}
COLS = {L: np.load(f"{REPO}/shared/astra_cols_L{L}.npy").astype(int) for L in LAYERS}
P = {}
for L in LAYERS:
    W = layers[L].mlp.down_proj.weight.float()                      # [1024, 3584]
    P[L] = U[L] @ W                                                  # u^T w_j, [3584]
rng = np.random.default_rng(0)
RAND = {L: np.sort(rng.choice(np.setdiff1d(np.arange(3584), COLS[L]), 16, replace=False)) for L in LAYERS}
TRACK = {L: np.concatenate([COLS[L], RAND[L]]) for L in LAYERS}       # 80 per layer
print({L: float(P[L][COLS[L]].mean()) for L in LAYERS}, "mean u^T w_j of Astra cols", flush=True)

# ---------------- corpus
def prep(r, think=True):
    ids, Pn = encode(tok, r["id"], r["text"], think=think, maxt=MAXT)
    m = loop_mask(ids); m[:Pn] = False
    return dict(id=r["id"], src=r["src"], ids=ids, P=Pn, m=m, loop_frac=float(m[Pn:].mean()) if len(ids) > Pn else 0.0)

ctxs = []
cut = [r for ds in ["gsm8k", "math500"] for r in astra_rows("unedited", ds) if "</think>" not in r["text"]]
cut_by = {"gsm8k": [r for r in cut if r["src"] == "gsm8k"], "math": [r for r in cut if r["src"] == "math"]}
loop = []
for k in range(max(len(v) for v in cut_by.values())):          # alternate gsm8k / math
    for s in ["gsm8k", "math"]:
        if k < len(cut_by[s]) and len(loop) < NPC:
            c = prep(cut_by[s][k])
            if c["loop_frac"] >= 0.05: c["cls"] = "loop"; loop.append(c)
done = [r for ds in ["gsm8k", "math500"] for r in astra_rows("unedited", ds) if "</think>" in r["text"]]
normal = []
for r in done:
    c = prep(r); c["cls"] = "normal"; c["kind"] = "think_finished"; normal.append(c)
gen = f"{DATA_OUT}/gen_base_nothink.jsonl"
if os.path.exists(gen):
    G = [json.loads(l) for l in open(gen)]
    G = [g for g in G if not g["cut"] and g["loop_frac"] < 0.01]
    random.Random(0).shuffle(G)
    for g in G:
        if len(normal) >= NPC: break
        c = prep(g, think=False); c["cls"] = "normal"; c["kind"] = "nothink_finished"; normal.append(c)
nonmath = []
for ds in ["ifeval", "mmlupro1k"]:
    for r in astra_rows("unedited", ds)[: NPC // 2]:
        c = prep(r); c["cls"] = "nonmath"; nonmath.append(c)
ctxs = loop + normal + nonmath
if os.environ.get("PILOT"):
    ctxs = loop[:3] + normal[:3] + nonmath[:2]
print(f"contexts: loop {len(loop)}, normal {len(normal)} ({sum(c.get('kind')=='nothink_finished' for c in normal)} no-think), "
      f"nonmath {len(nonmath)}; MAXT {MAXT}; mean len {np.mean([len(c['ids']) for c in ctxs]):.0f}", flush=True)

acts = {}
def mk(L):
    def h(mod, inp): acts[L] = inp[0][0]
    return h
hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in LAYERS]

def peaks(v, mask_ok, sep=64, k=8):
    """indices of up to k highest values of v (numpy) at mask_ok positions, >= sep apart."""
    order = np.argsort(-v); out = []
    for i in order[:2000]:
        if not mask_ok[i]: continue
        if all(abs(i - j) >= sep for j in out): out.append(int(i))
        if len(out) >= k: break
    return out

stats = {L: dict(sum_loop=[], sum_non=[], sq_loop=[], sq_non=[], abs_loop=[], abs_non=[]) for L in LAYERS}
meta, windows, rwindows, states, series = [], [], [], [], []
t0 = time.time()
with torch.no_grad():
    for ci, c in enumerate(ctxs):
        ids, Pn, m = c["ids"], c["P"], c["m"]
        out = model.model(torch.tensor([ids], device=DEV), output_hidden_states=True)
        T = len(ids); gen_ok = np.arange(T) >= Pn
        ml = torch.tensor(m & gen_ok, device=DEV); mn = torch.tensor((~m) & gen_ok, device=DEV)
        rec = dict(id=c["id"], src=c["src"], cls=c["cls"], kind=c.get("kind"), T=T, P=Pn, n_loop=int(ml.sum()), n_non=int(mn.sum()),
                   loop_frac=c["loop_frac"])
        ser = {}
        for L in LAYERS:
            a = acts[L].float()                                       # [T, 3584]
            cc = a * P[L]                                             # signed write along u
            st = stats[L]
            st["sum_loop"].append(cc[ml].sum(0).cpu()); st["sum_non"].append(cc[mn].sum(0).cpu())
            st["sq_loop"].append((cc[ml] ** 2).sum(0).cpu()); st["sq_non"].append((cc[mn] ** 2).sum(0).cpu())
            st["abs_loop"].append(a[ml].abs().sum(0).cpu()); st["abs_non"].append(a[mn].abs().sum(0).cpu())
            ser[f"resid_u_{L + 1}"] = (out.hidden_states[L + 1][0].float() @ U[L]).half().cpu()
            ser[f"resid_u_{L}"] = (out.hidden_states[L][0].float() @ U[L]).half().cpu()
            ser[f"write_astra_{L}"] = cc[:, COLS[L]].sum(1).half().cpu()
            ser[f"write_all_{L}"] = cc.sum(1).half().cpu()
            # exemplar peaks for tracked neurons
            ct = cc[:, TRACK[L]].cpu().numpy()                        # [T, 80]
            for k, j in enumerate(TRACK[L]):
                seen = set()
                for p in peaks(ct[:, k], gen_ok, k=6):
                    key = tuple(ids[max(0, p - 3): p + 4])
                    if key in seen: continue
                    seen.add(key)
                    s, e = max(0, p - W_BEFORE), min(T, p + W_AFTER + 1)
                    windows.append(dict(layer=L, unit=int(j), ctx=ci, pos=p, val=float(ct[p, k]), in_loop=bool(m[p]),
                                        ids=ids[s:e], acts=[round(float(x), 4) for x in ct[s:e, k]], peak=p - s))
            ser[f"track_{L}"] = None
        # random windows: 3 loop + 3 non-loop positions (if available), acts of all tracked neurons
        R = np.random.default_rng(ci)
        cand = [np.where(m & gen_ok)[0], np.where((~m) & gen_ok)[0]]
        cts = {L: (acts[L].float()[:, TRACK[L]] * P[L][TRACK[L]]).cpu().numpy() for L in LAYERS}
        for pool in cand:
            for p in (R.choice(pool, min(3, len(pool)), replace=False) if len(pool) else []):
                p = int(p); s, e = max(0, p - W_BEFORE), min(T, p + W_AFTER + 1)
                rwindows.append(dict(ctx=ci, pos=p, in_loop=bool(m[p]), ids=ids[s:e], peak=p - s,
                                     acts={L: np.round(cts[L][s:e], 4).astype(np.float16) for L in LAYERS}))
        # probe-training states (math contexts only)
        if c["cls"] in ("loop", "normal"):
            lp = np.where(m & gen_ok)[0]; npos = np.where((~m) & gen_ok)[0]
            lp = R.choice(lp, min(256, len(lp)), replace=False) if len(lp) else lp
            npos = R.choice(npos, min(256, len(npos)), replace=False)
            idx = np.sort(np.concatenate([lp, npos])).astype(int)
            states.append(dict(ctx=ci, pos=idx, is_loop=m[idx], H={L: out.hidden_states[L][0][idx].half().cpu() for L in PROBE_LAYERS}))
        ser.pop("track_15", None); ser.pop("track_19", None)
        series.append(dict(ctx=ci, m=m, **ser))
        meta.append(rec)
        del out
        if ci % 10 == 0 or os.environ.get("PILOT"):
            print(f"[{ci + 1}/{len(ctxs)}] {c['cls']} {c['src']} T={T} loop_frac={c['loop_frac']:.3f} "
                  f"resid_u16 loop={series[-1]['resid_u_16'][ml.cpu()].float().mean():.3f} non={series[-1]['resid_u_16'][mn.cpu()].float().mean():.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

os.makedirs(DATA_OUT, exist_ok=True)
outp = f"{DATA_OUT}/t3_collect_{MAXT}{'_pilot' if os.environ.get('PILOT') else ''}.pt"
torch.save(dict(meta=meta, ids=[c["ids"] for c in ctxs], stats={L: {k: torch.stack(v) for k, v in stats[L].items()} for L in LAYERS},
                P={L: P[L].cpu() for L in LAYERS}, cols=COLS, rand=RAND, track=TRACK, windows=windows, rwindows=rwindows,
                states=states, series=series, MAXT=MAXT), outp)
print("saved", outp, f"{time.time() - t0:.0f}s", flush=True)
