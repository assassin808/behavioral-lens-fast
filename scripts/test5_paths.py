"""Test 5: attribution + exact path patching of the p50c edit (L15/L19 down_proj, 64 cols) at loop positions.

Metric M = mean NLL of the actual next token at loop positions (32-gram seen before), per text; mean over texts.
For every downstream component X (heads of mixers L16-23, mixer blocks, MLP blocks L15-23) and groups:
  inject_X  base run, X <- p50c activation      (sufficiency, all paths from X)
  block_X   p50c run, X <- base activation      (necessity)
  direct_X  base final resid + Delta_X, re-normed, unembedded (last hop only)
  AtP_X     grad_X M . (a_X^p50c - a_X^base) with grads at base and at p50c
  controls  inject Gaussian noise with Delta_X's per-position norm / position-permuted Delta_X
Plus: copy (induction) attention of full-attn heads L15/19/23, logit lens of top mediators' Delta, L15 h7 sanity.

    python scripts/test5_paths.py N MAXT TAG        (out: notes/test5_paths_{TAG}.json, $T5_DATA/test5_{TAG}.pt)
"""
import glob, gzip, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
TAG = sys.argv[3] if len(sys.argv) > 3 else f"n{N}_t{MAXT}"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("BLF_MODELS", "/scratch/yang0531/blf_models")
MB = os.environ.get("BLF_MODEL_BASE", f"{MODELS}/base"); ME = os.environ.get("BLF_MODEL_P50C", f"{MODELS}/p50c")
DATA = os.environ.get("BLF_DATA") or os.environ.get("T5_DATA", "/home/yang0531/.claude/jobs/dcb4a01d/tmp/t5data")
os.makedirs(DATA, exist_ok=True)
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
torch.manual_seed(0); np.random.seed(0)
t0 = time.time()

tok = AutoTokenizer.from_pretrained(MB)
model = AutoModelForCausalLM.from_pretrained(MB, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
inner = model.model; layers = inner.layers; lm_head = model.lm_head; fnorm = inner.norm
FULL = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]
print("full-attn layers", FULL, flush=True)

# ---- edit weights; verify only L15/L19 down_proj differ
def st(d):
    fs = sorted(glob.glob(f"{d}/*.safetensors")); assert len(fs) == 1, fs; return load_file(fs[0])
sb, se = st(MB), st(ME)
diff = [k for k in sb if not torch.equal(sb[k], se[k])]
print("tensors that differ:", diff, flush=True)
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"
assert sorted(diff) == sorted(NAME.format(L) for L in (15, 19)), diff
W = {L: {"base": sb[NAME.format(L)].cuda(), "edit": se[NAME.format(L)].cuda()} for L in (15, 19)}
COLS = {L: torch.nonzero((W[L]["edit"] - W[L]["base"]).abs().sum(0) > 0).flatten() for L in (15, 19)}
U = {L: torch.tensor(np.load(f"{REPO}/shared/u_L{L}.npy"), dtype=torch.float32, device="cuda") for L in (15, 19)}
del sb, se
def set_weights(which):  # which: dict L -> 'base'|'edit'
    for L, w in which.items(): layers[L].mlp.down_proj.weight.data.copy_(W[L][w])
WB, WE = {15: "base", 19: "base"}, {15: "edit", 19: "edit"}

# ---- texts
qs = {}
for l in open(f"{REPO}/data/gsm8k_test.jsonl"): r = json.loads(l); qs[r["id"]] = r["question"]
for l in open(f"{REPO}/data/math500.jsonl"): r = json.loads(l); qs[r["id"]] = r["problem"]
rows = []
for ds in ("gsm8k", "math500"):
    for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/unedited/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
rng = np.random.default_rng(0); rows = [rows[i] for i in rng.permutation(len(rows))]
NG = 32
def loop_src(ids):
    first, src = {}, np.full(len(ids), -1)
    for i in range(NG, len(ids)):
        g = tuple(ids[i - NG:i])
        if g in first: src[i] = first[g]
        else: first[g] = i
    return src
texts = []
for r in rows:
    if "</think>" in r["text"] or r["id"] not in qs: continue
    q = qs[r["id"]] + "\n" + INSTR
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; gid = tok(r["text"], add_special_tokens=False).input_ids
    ids = (pid + gid)[:MAXT]
    if len(ids) < MAXT: continue
    src = loop_src(ids); src[:len(pid)] = -1
    if (src >= 0).mean() > 0.03: texts.append((r["id"], ids, src, len(pid)))
    if len(texts) >= N: break
print(f"{len(texts)} texts x {MAXT} tok; loop_frac={[round(float((s>=0).mean()),3) for _,_,s,_ in texts]}", flush=True)

# ---- components
COMP = {}   # name -> (kind, layer, slice or None)
for L in range(16, 24):
    nh, hd = (8, 256) if L in FULL else (16, 128)
    for h in range(nh): COMP[f"h{L}.{h}"] = ("z", L, slice(h * hd, (h + 1) * hd))
    COMP[f"attn{L}"] = ("z", L, slice(None))
for L in range(15, 24): COMP[f"mlp{L}"] = ("mlp", L, None)
GROUPS = {
    "attn16-23": [f"attn{L}" for L in range(16, 24)], "attn16-21": [f"attn{L}" for L in range(16, 22)],
    "attn22-23": ["attn22", "attn23"], "fullattn19+23": ["attn19", "attn23"],
    "mlp16-23(excl19)": [f"mlp{L}" for L in range(16, 24) if L != 19], "mlp20-23": [f"mlp{L}" for L in range(20, 24)],
    "mlp16-18": ["mlp16", "mlp17", "mlp18"], "mlp20-21": ["mlp20", "mlp21"], "mlp22-23": ["mlp22", "mlp23"],
    "all_downstream(excl mlp19)": [f"attn{L}" for L in range(16, 24)] + [f"mlp{L}" for L in range(16, 24) if L != 19],
    "layers22-23": ["attn22", "attn23", "mlp22", "mlp23"],
}
def mixer(L):
    return layers[L].self_attn.o_proj if L in FULL else layers[L].linear_attn.out_proj
WO = {L: mixer(L).weight for L in range(15, 24)}   # [1024, 2048]

# ---- hooks
H = dict(mode=None, cache={}, patch={}, grads={}, keep={})
# patch: key ('z',L) or ('mlp',L) -> list of (slice, tensor[T,D], mode 'rep'|'add', posmask or None)
def apply_patch(key, x):
    pl = H["patch"].get(key)
    if not pl: return x
    x = x.clone()
    for sl, t, mode, pm in pl:
        sl = sl if sl is not None else slice(None)
        if pm is None:
            if mode == "rep": x[0, :, sl] = t[:, sl].to(x.dtype)
            else: x[0, :, sl] = x[0, :, sl] + t[:, sl].to(x.dtype)
        else:
            if mode == "rep": x[0, pm, sl] = t[pm][:, sl].to(x.dtype)
            else: x[0, pm, sl] = x[0, pm, sl] + t[pm][:, sl].to(x.dtype)
    return x
def mk_pre(L):
    def f(mod, args):
        z = apply_patch(("z", L), args[0])
        if H["mode"] == "cache": H["cache"][("z", L)] = z[0].detach()
        if H["mode"] == "grad": z.retain_grad(); H["keep"][("z", L)] = z
        return (z,) + tuple(args[1:])
    return f
def mk_mlp(L):
    def f(mod, args, out):
        o = apply_patch(("mlp", L), out)
        if H["mode"] == "cache": H["cache"][("mlp", L)] = o[0].detach()
        if H["mode"] == "grad": o.retain_grad(); H["keep"][("mlp", L)] = o
        return o
    return f
def mk_dp15(mod, args):   # L15 down_proj input = neuron activations
    if H["mode"] == "cache": H["cache"][("act15",)] = args[0][0, :, COLS[15]].detach()
def pre_norm(mod, args):
    if H["mode"] in ("cache",): H["cache"][("final",)] = args[0][0].detach()
for L in range(15, 24):
    mixer(L).register_forward_pre_hook(mk_pre(L)); layers[L].mlp.register_forward_hook(mk_mlp(L))
layers[15].mlp.down_proj.register_forward_pre_hook(mk_dp15)
fnorm.register_forward_pre_hook(pre_norm)
def emb_grad(mod, args, out):
    if H["mode"] == "grad": return out.detach().requires_grad_(True)
inner.embed_tokens.register_forward_hook(emb_grad)

def metric_from_hidden(h, ids_t, lp, pp):
    out = []
    for pos in (lp, pp):
        lg = lm_head(h[pos]).float()
        out.append(F.cross_entropy(lg, ids_t[pos + 1], reduction="none").mean())
    return out
def run(ids_t, lp, pp, weights, patch=None, mode=None, attn=False):
    set_weights(weights); H["mode"] = mode; H["patch"] = patch or {}; H["cache"] = {} if mode == "cache" else H["cache"]
    H["keep"] = {}
    ctx = torch.enable_grad() if mode == "grad" else torch.no_grad()
    with ctx:
        o = inner(ids_t[None], output_attentions=attn)
        ml, mp = metric_from_hidden(o.last_hidden_state[0], ids_t, lp, pp)
        if mode == "grad":
            ml.backward()
            g = {k: v.grad[0].detach().float() for k, v in H["keep"].items()}
            H["keep"] = {}
            return ml.item(), mp.item(), g
    H["mode"] = None; H["patch"] = {}
    return (ml.item(), mp.item(), o.attentions) if attn else (ml.item(), mp.item())

def comp_patch(names, src, mode="rep", tens=None, pm=None):
    """build patch dict: replace components `names` with values from cache `src` (or add `tens[name]`)."""
    P = {}
    for n in names:
        kind, L, sl = COMP[n]; key = (kind, L)
        t = src[key] if tens is None else tens[n]
        P.setdefault(key, []).append((sl, t, mode, pm))
    return P
def resid_delta(n, cb, ce):
    kind, L, sl = COMP[n]
    if kind == "mlp": return (ce[("mlp", L)] - cb[("mlp", L)]).float()
    dz = (ce[("z", L)] - cb[("z", L)]).float(); m = torch.zeros_like(dz); m[:, sl] = dz[:, sl]
    return m @ WO[L].float().T
def act_delta(n, cb, ce):
    kind, L, sl = COMP[n]; key = (kind, L)
    d = (ce[key] - cb[key]).float()
    if kind == "z": m = torch.zeros_like(d); m[:, sl] = d[:, sl]; d = m
    return d

STOP = ["</think>", "\n\n", " So", " Therefore", " Wait", "Wait", " final", " answer", "\\boxed", " Thus", " Let", " Hmm", " But",
        " first", " result", " value", "**", " Final", "Final", " The", "So"]
STOP_IDS = {s: tok(s, add_special_tokens=False).input_ids for s in STOP}
STOP_IDS = {s: v[0] for s, v in STOP_IDS.items() if len(v) == 1}

R = []   # per-text results
LENS_SUM = {}  # name -> sum over loop positions of centered dlogit vectors (cpu float)
LENS_N = 0
for ti, (tid, ids, src, P) in enumerate(texts):
    ids_t = torch.tensor(ids, device="cuda"); T = len(ids)
    pos = np.arange(T - 1); isl = (src[1:] >= 0) & (pos >= P)
    lp = torch.tensor(pos[isl], device="cuda"); pp = torch.tensor(pos[(~isl) & (pos >= P)], device="cuda")
    pm_loop = torch.zeros(T, dtype=torch.bool, device="cuda"); pm_loop[lp] = True
    rec = {"id": tid, "T": T, "n_loop": int(len(lp)), "n_prod": int(len(pp))}
    # caches
    Mb = run(ids_t, lp, pp, WB, mode="cache"); cb = H["cache"]; H["cache"] = {}
    Me = run(ids_t, lp, pp, WE, mode="cache"); ce = H["cache"]; H["cache"] = {}
    rec["base"], rec["edit"] = Mb, Me
    rec["edit15only"] = run(ids_t, lp, pp, {15: "edit", 19: "base"}); rec["edit19only"] = run(ids_t, lp, pp, {15: "base", 19: "edit"})
    # identity checks
    rec["id_base"] = run(ids_t, lp, pp, WB, comp_patch(["attn20", "mlp22"], cb))
    rec["id_edit"] = run(ids_t, lp, pp, WE, comp_patch(["attn20", "mlp22"], ce))
    # AtP
    mlb, _, gb = run(ids_t, lp, pp, WB, mode="grad"); mle, _, ge = run(ids_t, lp, pp, WE, mode="grad")
    rec["atp_base"], rec["atp_edit"] = {}, {}
    for n, (kind, L, sl) in COMP.items():
        d = act_delta(n, cb, ce)
        rec["atp_base"][n] = float((gb[(kind, L)] * d).sum()); rec["atp_edit"][n] = float((ge[(kind, L)] * d).sum())
    del gb, ge
    # exact inject / block, single components
    rec["inject"], rec["block"], rec["direct"] = {}, {}, {}
    fb = cb[("final",)].float()
    def direct_metric(delta):
        h = fnorm((fb + delta).to(torch.bfloat16)[None])[0]
        with torch.no_grad(): return [m.item() for m in metric_from_hidden(h, ids_t, lp, pp)]
    with torch.no_grad():
        for n in COMP:
            rec["inject"][n] = run(ids_t, lp, pp, WB, comp_patch([n], ce))
            rec["block"][n] = run(ids_t, lp, pp, WE, comp_patch([n], cb))
            rec["direct"][n] = direct_metric(resid_delta(n, cb, ce))
        rec["direct"]["ALL"] = direct_metric(ce[("final",)].float() - fb)
        rec["direct"]["edit_sites(mlp15+mlp19)"] = direct_metric(resid_delta("mlp15", cb, ce) + resid_delta("mlp19", cb, ce))
        for g, names in GROUPS.items():
            rec["inject"][g] = run(ids_t, lp, pp, WB, comp_patch(names, ce))
            rec["block"][g] = run(ids_t, lp, pp, WE, comp_patch(names, cb))
        # loop-position-only vs non-loop-only injection, layer blocks
        rec["inject_looppos"], rec["inject_nonlooppos"], rec["ctrl_noise"], rec["ctrl_perm"] = {}, {}, {}, {}
        ctrl_names = [f"attn{L}" for L in range(16, 24)] + [f"mlp{L}" for L in range(15, 24)] + \
                     [f"h{L}.{h}" for L in (19, 23) for h in range(8)] + list(GROUPS)
        for n in ctrl_names:
            names = GROUPS.get(n, [n])
            if n not in GROUPS:
                rec["inject_looppos"][n] = run(ids_t, lp, pp, WB, comp_patch(names, ce, pm=pm_loop))
                rec["inject_nonlooppos"][n] = run(ids_t, lp, pp, WB, comp_patch(names, ce, pm=~pm_loop))
            tn, tp = {}, {}
            perm = torch.randperm(T, device="cuda")
            for m in names:
                d = act_delta(m, cb, ce)                       # zero outside the head's slice
                z = torch.randn_like(d) * (d.abs().sum(0, keepdim=True) > 0)
                z = z * (d.norm(dim=1, keepdim=True) / (z.norm(dim=1, keepdim=True) + 1e-8))
                tn[m] = z; tp[m] = d[perm]
            rec["ctrl_noise"][n] = run(ids_t, lp, pp, WB, comp_patch(names, None, "add", tn))
            rec["ctrl_perm"][n] = run(ids_t, lp, pp, WB, comp_patch(names, None, "add", tp))
        # copy attention (full attn layers 15, 19, 23): query p (predicts ids[p+1]); target j* = src[p+1] holds the repeated token
        rec["copy"] = {}
        jt = torch.tensor(src[1:][isl], device="cuda")   # = src[p+1]
        for tagw, ww in (("base", WB), ("edit", WE)):
            _, _, atts = run(ids_t, lp, pp, ww, attn=True)
            for k, L in enumerate(FULL):
                if L not in (15, 19, 23): continue
                A = (atts[k] if len(atts) == len(FULL) else atts[L])[0]   # [H, T, T]
                ind = A[:, lp, jt].float().mean(1); dup = A[:, lp, (jt - 1).clamp(min=0)].float().mean(1)
                rec["copy"][f"{tagw}_L{L}"] = dict(induction=ind.tolist(), duplicate=dup.tolist())
            del atts
        # L15 h7 sanity: ablate (zero its o_proj input slice == Astra's zeroed o_proj cols 1792:2048)
        z0 = {("z", 15): [(slice(1792, 2048), torch.zeros(T, 2048, device="cuda"), "rep", None)]}
        H["patch"] = z0
        rec["h7_ablate_base"] = run(ids_t, lp, pp, WB, z0, mode="cache"); a_abl = H["cache"][("act15",)]; H["cache"] = {}
        rec["h7_ablate_edit"] = run(ids_t, lp, pp, WE, z0)
        rec["h7_ctrl_ablate_h3_base"] = run(ids_t, lp, pp, WB, {("z", 15): [(slice(768, 1024), torch.zeros(T, 2048, device="cuda"), "rep", None)]})
        uw = (U[15] @ W[15]["base"][:, COLS[15]].float())                    # u^T w_j (base)
        def uwrite(act, m): return float((act[m].float() * uw).sum(1).mean())
        lpos = torch.tensor(np.where(src >= 0)[0], device="cuda"); ppos = torch.tensor(np.where((src < 0) & (np.arange(T) >= P))[0], device="cuda")
        rec["h7_uwrite"] = dict(base_loop=uwrite(cb[("act15",)], lpos), abl_loop=uwrite(a_abl, lpos),
                                base_prod=uwrite(cb[("act15",)], ppos), abl_prod=uwrite(a_abl, ppos))
        zh7 = cb[("z", 15)][:, 1792:2048].float() @ WO[15][:, 1792:2048].float().T
        rec["h7_out_cos_u15_loop"] = float(F.cosine_similarity(zh7[lpos], U[15][None], dim=1).mean())
        rec["h7_out_cos_u15_prod"] = float(F.cosine_similarity(zh7[ppos], U[15][None], dim=1).mean())
        rec["h7_out_u15_proj_loop"] = float((zh7[lpos] @ U[15]).mean()); rec["h7_out_u15_proj_prod"] = float((zh7[ppos] @ U[15]).mean())
        # logit lens of Delta_X (direct, linearised through the base final-norm scale) at loop positions
        rms = fb[lp].pow(2).mean(1, keepdim=True).add(1e-6).rsqrt(); gam = (1 + fnorm.weight.float())
        WU = lm_head.weight.float()
        rec["lens"] = {}
        lens_names = [f"attn{L}" for L in range(16, 24)] + [f"mlp{L}" for L in range(15, 24)] + [f"h{L}.{h}" for L in (19, 23) for h in range(8)]
        tgt = ids_t[lp + 1]
        for n in lens_names + ["ALL"]:
            d = (ce[("final",)].float() - fb) if n == "ALL" else resid_delta(n, cb, ce)
            d = d[lp]
            tgs, pcts, sums = [], [], 0
            for c0 in range(0, len(lp), 256):
                dl = (d[c0:c0+256] * rms[c0:c0+256] * gam) @ WU.T                 # [chunk, V]
                dlc = dl - dl.mean(1, keepdim=True); del dl
                tg = dlc.gather(1, tgt[c0:c0+256, None])[:, 0]
                pcts.append((dlc < tg[:, None]).float().mean(1)); tgs.append(tg)
                sums = sums + dlc.sum(0); stopv = dlc[:, list(STOP_IDS.values())].sum(0) if c0 == 0 else stopv + dlc[:, list(STOP_IDS.values())].sum(0)
                del dlc
            tg = torch.cat(tgs); pct = torch.cat(pcts)
            mean_d = d.mean(0)
            consist = float(F.cosine_similarity(d, mean_d[None], dim=1).mean())
            rec["lens"][n] = dict(target_centered=float(tg.mean()), target_pct_median=float(pct.median()), consist=consist,
                                  stop={s: float(stopv[k] / len(lp)) for k, s in enumerate(STOP_IDS)},
                                  dnorm=float(d.norm(dim=1).mean()), cos_u15=float(F.cosine_similarity(d, U[15][None], dim=1).mean()),
                                  cos_u19=float(F.cosine_similarity(d, U[19][None], dim=1).mean()))
            LENS_SUM[n] = LENS_SUM.get(n, 0) + sums.cpu()
        LENS_N += len(lp)
    R.append(rec)
    tot = Me[0] - Mb[0]
    print(f"[{ti}] {tid} T={T} nloop={len(lp)} base={Mb[0]:.4f} edit={Me[0]:.4f} d={tot:+.4f} prod {Mb[1]:.3f}->{Me[1]:.3f} "
          f"id_err=({rec['id_base'][0]-Mb[0]:+.1e},{rec['id_edit'][0]-Me[0]:+.1e}) gradM=({mlb:.4f},{mle:.4f}) "
          f"h7abl={rec['h7_ablate_base'][0]-Mb[0]:+.4f} t={time.time()-t0:.0f}s", flush=True)
    if ti == 0 or ti == len(texts) - 1:   # eyeball raw predictions at a few loop positions
        with torch.no_grad():
            for w, nm in ((WB, "base"), (WE, "edit")):
                set_weights(w); hh = inner(ids_t[None]).last_hidden_state[0]
                for p in lp[:: max(1, len(lp) // 4)][:4].tolist():
                    pr = F.softmax(lm_head(hh[p]).float(), -1); top = pr.topk(5)
                    print(f"   {nm} pos {p} ctx=...{tok.decode(ids[max(P,p-12):p+1])!r} target={tok.decode([ids[p+1]])!r} "
                          f"p={pr[ids[p+1]].item():.3f} top={[(tok.decode([i]), round(v,3)) for v,i in zip(top.values.tolist(), top.indices.tolist())]}", flush=True)
    del cb, ce; torch.cuda.empty_cache()

# ---- aggregate
def agg(key, sub=None):
    vals = [r[key] if sub is None else r[key][sub] for r in R]
    return np.array(vals)
base = agg("base")[:, 0]; edit = agg("edit")[:, 0]; tot = float((edit - base).mean())
S = dict(n_texts=len(R), maxt=MAXT, n_loop_pos=int(sum(r["n_loop"] for r in R)), ids=[r["id"] for r in R],
         M_base=float(base.mean()), M_edit=float(edit.mean()), total=tot,
         prod_base=float(agg("base")[:, 1].mean()), prod_edit=float(agg("edit")[:, 1].mean()),
         edit15only=float(agg("edit15only")[:, 0].mean() - base.mean()), edit19only=float(agg("edit19only")[:, 0].mean() - base.mean()),
         id_err=float(np.abs(np.r_[agg("id_base")[:, 0] - base, agg("id_edit")[:, 0] - edit]).max()))
def boot(x, B=2000):
    i = np.random.default_rng(1).integers(0, len(x), (B, len(x))); m = x[i].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
comp = {}
allnames = list(COMP) + list(GROUPS)
for n in allnames:
    c = {}
    inj = np.array([r["inject"][n][0] for r in R]) - base; blk = edit - np.array([r["block"][n][0] for r in R])
    c["inject"], c["block"] = float(inj.mean()), float(blk.mean())
    c["inject_ci"], c["block_ci"] = boot(inj), boot(blk)
    c["inject_frac"], c["block_frac"] = c["inject"] / tot, c["block"] / tot
    if n in COMP:
        c["direct"] = float((np.array([r["direct"][n][0] for r in R]) - base).mean())
        c["atp_base"] = float(np.mean([r["atp_base"][n] for r in R])); c["atp_edit"] = float(np.mean([r["atp_edit"][n] for r in R]))
        c["atp_mean"] = 0.5 * (c["atp_base"] + c["atp_edit"])
    for k in ("inject_looppos", "inject_nonlooppos", "ctrl_noise", "ctrl_perm"):
        if n in R[0][k]: c[k] = float((np.array([r[k][n][0] for r in R]) - base).mean())
    comp[n] = c
S["direct_ALL"] = float((np.array([r["direct"]["ALL"][0] for r in R]) - base).mean())
S["direct_edit_sites"] = float((np.array([r["direct"]["edit_sites(mlp15+mlp19)"][0] for r in R]) - base).mean())
S["comp"] = comp
S["copy"] = {k: {kk: np.mean([r["copy"][k][kk] for r in R], 0).tolist() for kk in ("induction", "duplicate")} for k in R[0]["copy"]}
S["h7"] = dict(ablate_base=float(agg("h7_ablate_base")[:, 0].mean() - base.mean()),
               ablate_base_prod=float(agg("h7_ablate_base")[:, 1].mean() - agg("base")[:, 1].mean()),
               ablate_edit_extra=float(agg("h7_ablate_edit")[:, 0].mean() - edit.mean()),
               ctrl_h3_ablate_base=float(agg("h7_ctrl_ablate_h3_base")[:, 0].mean() - base.mean()),
               uwrite={k: float(np.mean([r["h7_uwrite"][k] for r in R])) for k in R[0]["h7_uwrite"]},
               out_cos_u15_loop=float(np.mean([r["h7_out_cos_u15_loop"] for r in R])), out_cos_u15_prod=float(np.mean([r["h7_out_cos_u15_prod"] for r in R])),
               out_u15_proj_loop=float(np.mean([r["h7_out_u15_proj_loop"] for r in R])), out_u15_proj_prod=float(np.mean([r["h7_out_u15_proj_prod"] for r in R])))
S["lens"] = {}
for n in R[0]["lens"]:
    L_ = {k: float(np.mean([r["lens"][n][k] for r in R])) for k in ("target_centered", "target_pct_median", "consist", "dnorm", "cos_u15", "cos_u19")}
    L_["stop"] = {s: float(np.mean([r["lens"][n]["stop"][s] for r in R])) for s in R[0]["lens"][n]["stop"]}
    mv = LENS_SUM[n] / LENS_N; top = mv.topk(15); bot = (-mv).topk(15)
    L_["promoted"] = [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
    L_["suppressed"] = [(tok.decode([i]), round(-v, 3)) for v, i in zip(bot.values.tolist(), bot.indices.tolist())]
    S["lens"][n] = L_
from scipy.stats import spearmanr
sn = [n for n in COMP]
S["atp_vs_inject_spearman"] = dict(base=float(spearmanr([comp[n]["atp_base"] for n in sn], [comp[n]["inject"] for n in sn])[0]),
                                   edit=float(spearmanr([comp[n]["atp_edit"] for n in sn], [comp[n]["inject"] for n in sn])[0]),
                                   mean=float(spearmanr([comp[n]["atp_mean"] for n in sn], [comp[n]["inject"] for n in sn])[0]))
json.dump(S, open(f"{REPO}/notes/test5_paths_{TAG}.json", "w"), indent=1)
torch.save(R, f"{DATA}/test5_{TAG}.pt")
print(f"\nn={len(R)} texts x {MAXT} tok, {S['n_loop_pos']} loop positions; M base {S['M_base']:.4f} edit {S['M_edit']:.4f} total {tot:+.4f}; "
      f"L15-only {S['edit15only']:+.4f} L19-only {S['edit19only']:+.4f}; id_err {S['id_err']:.1e}")
print(f"direct ALL {S['direct_ALL']:+.4f}  direct edit sites {S['direct_edit_sites']:+.4f}  AtP spearman {S['atp_vs_inject_spearman']}")
print(f"{'component':28s} {'inject':>8s} {'block':>8s} {'inj%':>6s} {'blk%':>6s} {'direct':>8s} {'AtPb':>8s} {'AtPe':>8s} {'noise':>8s} {'perm':>8s}")
for n in sorted(allnames, key=lambda n: -abs(comp[n]["block"]))[:45]:
    c = comp[n]
    print(f"{n:28s} {c['inject']:+8.4f} {c['block']:+8.4f} {100*c['inject_frac']:6.1f} {100*c['block_frac']:6.1f} "
          f"{c.get('direct', float('nan')):+8.4f} {c.get('atp_base', float('nan')):+8.4f} {c.get('atp_edit', float('nan')):+8.4f} "
          f"{c.get('ctrl_noise', float('nan')):+8.4f} {c.get('ctrl_perm', float('nan')):+8.4f}")
print("copy attention:", json.dumps({k: {kk: [round(x, 3) for x in vv] for kk, vv in v.items()} for k, v in S["copy"].items()}))
print("h7:", json.dumps(S["h7"]))
for n in ["ALL", "mlp15", "mlp19", "mlp22", "mlp23", "attn22", "attn23"]:
    l = S["lens"][n]; print(f"lens {n}: tgt {l['target_centered']:+.3f} pct {l['target_pct_median']:.3f} consist {l['consist']:.2f} "
                           f"cos_u15 {l['cos_u15']:+.2f}\n   + {l['promoted']}\n   - {l['suppressed']}\n   stop {json.dumps({k: round(v,2) for k,v in l['stop'].items()})}")
print(f"done {time.time()-t0:.0f}s")
