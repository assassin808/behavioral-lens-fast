"""Test 4: which SAE features are u, which does the edit turn off / on; exemplars for auto-interp.

    python scripts/test4_analyze.py ACTDIR SAE16.pt SAE20.pt OUT.json [N_LOOP=40] [MAXT=8192]

SAEs: ours16 (hidden_states[16]), pub18 (decoderesearch layer-17 = hidden_states[18]), ours20 (hidden_states[20]).
1. cos(decoder, u15/u19) and per-feature contribution to h·u at loop vs productive positions (collected base acts, train split).
2. Base loop-vs-productive selectivity of every feature (train split, 400k loop + 400k productive generated positions).
3. Paired base / p50c on held-out looping generations: mean activation change at loop and productive positions;
   cos(decoder, mean Δresid at loop positions).
4. Exemplars for selected features: top-activating contexts (train split, <= 2 per generation id, <= 1 per 64-token window)
   and held-out scoring sets (test split: 6 activating from activation quantiles + 6 random non-activating).
"""
import sys, json, time, numpy as np, torch
from test24_common import *
ACT, S16, S20, OUT = sys.argv[1:5]
NL = int(sys.argv[5]) if len(sys.argv) > 5 else 40; MAXT = int(sys.argv[6]) if len(sys.argv) > 6 else 8192
rng = np.random.default_rng(0); torch.manual_seed(0)
SAES = {"ours16": (16, SAEWrap("ours", S16)), "pub18": (18, SAEWrap("pub", f"{DATA}/ext/sae_l17/sae_weights.safetensors")),
        "ours20": (20, SAEWrap("ours", S20))}
U = {k: torch.tensor(v).cuda() for k, v in get_u().items()}
meta = np.load(f"{ACT}/meta.npz"); items = json.load(open(f"{ACT}/items.json")); TOK = np.load(f"{ACT}/tok.npy")
item, posn, loopf = meta["item"], meta["pos"], meta["loop"]
isgen = posn >= np.array([it["P"] for it in items])[item]; istest = np.array([it["split"] == "test" for it in items])[item]
ischat = np.array([it["ds"] != "wiki" for it in items])[item]
ok = posn >= 16
A = {L: np.load(f"{ACT}/acts_L{L}.npy", mmap_mode="r") for L in [16, 18, 20]}
R = {"meta": dict(act_tokens=int(len(item)), n_items=len(items))}

def batched(L, idx, f, bs=8192):
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.asarray(A[L][idx[s:s + bs]])).cuda().float()
        out.append(f(x))
    return out

# ---- 1+2: base statistics on train split
def block_sample(mask, n, block=4096):
    """positions where mask is true, taken from random contiguous blocks (fast memmap reads), until >= n."""
    starts = rng.permutation(len(mask) // block) * block; out, tot = [], 0
    for s0 in starts:
        w = np.flatnonzero(mask[s0:s0 + block]) + s0
        if len(w): out.append(w); tot += len(w)
        if tot >= n: break
    return np.sort(np.concatenate(out))[:n] if out else np.array([], int)
loop_idx = block_sample(loopf & ~istest & ok, 400000); prod_idx = block_sample(isgen & ~loopf & ~istest & ok & ischat, 400000)
wiki_idx = block_sample(~ischat & ~istest & ok, 200000)
print("stat positions loop/prod/wiki", len(loop_idx), len(prod_idx), len(wiki_idx), flush=True)
F = {}
for nm, (L, sae) in SAES.items():
    st = {}
    for g, idx in [("loop", loop_idx), ("prod", prod_idx), ("wiki", wiki_idx)]:
        s1 = torch.zeros(sae.m, device="cuda"); s2 = torch.zeros(sae.m, device="cuda"); fr = torch.zeros(sae.m, device="cuda")
        for x in batched(L, idx, lambda x: x):
            z = sae.encode(x); s1 += z.sum(0); s2 += (z ** 2).sum(0); fr += (z > 0).sum(0)
        n = len(idx); st[g] = dict(mean=s1 / n, var=s2 / n - (s1 / n) ** 2, freq=fr / n)
    uk = 15 if L < 20 else 19
    wu = sae.dirs @ U[uk] * sae.W_dec.norm(dim=1) * sae.c          # contribution of unit activation to h·u
    F[nm] = dict(L=L, uk=uk, cos_u=(sae.dirs @ U[uk]), cos_u15=(sae.dirs @ U[15]), st=st,
                 dprime=(st["loop"]["mean"] - st["prod"]["mean"]) / torch.sqrt(0.5 * (st["loop"]["var"] + st["prod"]["var"]) + 1e-6),
                 contrib_loop=st["loop"]["mean"] * wu, contrib_prod=st["prod"]["mean"] * wu)
    # how well does the dictionary explain u: best single cos, and OMP-style 5-feature fit
    cu = F[nm]["cos_u"]; print(nm, "max|cos(dec,u)|", float(cu.abs().max()), "freq-dead", float((st["loop"]["freq"] + st["prod"]["freq"] + st["wiki"]["freq"] == 0).float().mean()), flush=True)
    sel = []; r = U[uk].clone()
    for _ in range(10):
        j = int((sae.dirs @ r).abs().argmax()); sel.append(j)
        D = sae.dirs[sel].T; coef = torch.linalg.lstsq(D, U[uk][:, None]).solution; r = U[uk] - (D @ coef)[:, 0]
        F[nm].setdefault("omp_resid", []).append(float(r.norm()))
    F[nm]["omp_feats"] = sel

# ---- 3: paired base / edit on held-out looping generations
base, tok = load_model("base"); edit, _ = load_model("p50c")
def caps(model, ids):
    layers = lm_parts(model)[1]; cap = {}; hs = []
    for i in [15, 17, 19]:
        hs.append(layers[i].register_forward_hook(lambda mod, inp, out, i=i: cap.__setitem__(i + 1, (out[0] if isinstance(out, tuple) else out)[0].float())))
    with torch.no_grad(): model.model(torch.tensor(ids[None].astype(np.int64)).cuda())
    for h in hs: h.remove()
    return cap
chat = [x for x in chat_items(tok, maxt=MAXT) if x["split"] == "test"]
loopi = [x for x in chat if x["gen_by"] == "unedited" and x["loop"].mean() > 0.03]
loopi = [loopi[i] for i in rng.permutation(len(loopi))[:NL]]
D = {nm: {g: dict(b=0, e=0, n=0, fb=0, fe=0) for g in ["loop", "prod"]} for nm in SAES}
dres = {L: 0 for L in [16, 18, 20]}; ndres = 0; recon = {nm: [] for nm in SAES}
for k, x in enumerate(loopi):
    cb, ce = caps(base, x["ids"]), caps(edit, x["ids"])
    gen = np.arange(max(x["P"], 16), len(x["ids"])); lp = x["loop"][gen]
    for nm, (L, sae) in SAES.items():
        zb, ze = sae.encode(cb[L]), sae.encode(ce[L])
        if k < 10:
            for zz, hh, tg in [(zb, cb[L], "base"), (ze, ce[L], "edit")]:
                p = torch.tensor(gen[lp]).cuda()
                if len(p): recon[nm].append((tg, float(((sae.decode(zz[p]) - hh[p]) ** 2).sum() / ((hh[p] - hh[p].mean(0)) ** 2).sum())))
        for g, msk in [("loop", lp), ("prod", ~lp)]:
            p = torch.tensor(gen[msk]).cuda()
            if len(p) == 0: continue
            d = D[nm][g]; d["b"] += zb[p].sum(0); d["e"] += ze[p].sum(0); d["n"] += len(p); d["fb"] += (zb[p] > 0).sum(0); d["fe"] += (ze[p] > 0).sum(0)
    p = torch.tensor(gen[lp]).cuda()
    for L in dres: dres[L] = dres[L] + (ce[L][p] - cb[L][p]).sum(0)
    ndres += len(p)
    print("paired", k, len(x["ids"]), int(lp.sum()), flush=True)
R["paired"] = dict(n_items=len(loopi), maxt=MAXT, n_loop_pos=D["ours16"]["loop"]["n"], n_prod_pos=D["ours16"]["prod"]["n"])
R["recon_fvu_loop"] = {nm: {tg: float(np.mean([v for t, v in recon[nm] if t == tg])) for tg in ["base", "edit"]} for nm in SAES}
sel_all = {}
for nm, (L, sae) in SAES.items():
    f = F[nm]; dl, dp = D[nm]["loop"], D[nm]["prod"]
    mb, me = dl["b"] / dl["n"], dl["e"] / dl["n"]; diff = me - mb
    pdiff = dp["e"] / dp["n"] - dp["b"] / dp["n"]
    dr = dres[L] / ndres; cos_dr = sae.dirs @ (dr / dr.norm())
    prodtop = set(torch.topk(-f["dprime"], 100).indices.tolist())
    groups = {"cos_plus_u": torch.topk(f["cos_u"], 8).indices.tolist(), "cos_minus_u": torch.topk(-f["cos_u"], 5).indices.tolist(),
              "contrib_u_loop_minus_prod": torch.topk(f["contrib_loop"] - f["contrib_prod"], 8).indices.tolist(),
              "edit_off_loop": torch.topk(-diff, 10).indices.tolist(), "edit_on_loop": torch.topk(diff, 20).indices.tolist(),
              "cos_dresid": torch.topk(cos_dr, 5).indices.tolist(), "cos_minus_dresid": torch.topk(-cos_dr, 3).indices.tolist(),
              "loop_selective": torch.topk(f["dprime"], 5).indices.tolist(), "prod_selective": torch.topk(-f["dprime"], 5).indices.tolist()}
    on20 = groups["edit_on_loop"]
    def row(j):
        return dict(cos_u=float(f["cos_u"][j]), cos_u15=float(f["cos_u15"][j]), dprime=float(f["dprime"][j]),
                    freq_loop=float(f["st"]["loop"]["freq"][j]), freq_prod=float(f["st"]["prod"]["freq"][j]), freq_wiki=float(f["st"]["wiki"]["freq"][j]),
                    mean_loop=float(f["st"]["loop"]["mean"][j]), mean_prod=float(f["st"]["prod"]["mean"][j]),
                    contrib_u_loop=float(f["contrib_loop"][j]), contrib_u_prod=float(f["contrib_prod"][j]),
                    edit_base_loop=float(mb[j]), edit_edit_loop=float(me[j]), edit_diff_loop=float(diff[j]), edit_diff_prod=float(pdiff[j]),
                    edit_freq_base_loop=float(dl["fb"][j] / dl["n"]), edit_freq_edit_loop=float(dl["fe"][j] / dl["n"]), cos_dresid=float(cos_dr[j]))
    feats = sorted(set(sum(groups.values(), [])))
    R[nm] = dict(layer_hs=L, u=f"u{f['uk']}", d_sae=sae.m, cfg=sae.cfg, max_abs_cos_u=float(f["cos_u"].abs().max()),
                 n_feats_cos_gt_0p3=int((f["cos_u"].abs() > 0.3).sum()), omp_resid_norm=f["omp_resid"], omp_feats=f["omp_feats"],
                 cos_u_dresid=float((dr / dr.norm()) @ U[f["uk"]]), dresid_rel=float(dr.norm()),
                 frac_on20_in_prodtop100=float(np.mean([j in prodtop for j in on20])), groups=groups,
                 total_abs_diff_loop=float(diff.abs().sum()), total_abs_diff_prod=float(pdiff.abs().sum()),
                 feats={int(j): row(j) for j in feats})
    sel_all[nm] = feats
    print(nm, "max cos", R[nm]["max_abs_cos_u"], "on20 in prodtop100", R[nm]["frac_on20_in_prodtop100"], flush=True)
del base, edit; torch.cuda.empty_cache()

# ---- 4: exemplars
tr_idx = te_idx = None
def ctx(i, before=40, after=6):
    it = item[i]; s = i - min(before, posn[i]); e = i + 1
    while e < len(item) and item[e] == it and e < i + 1 + after: e += 1
    return dict(id=items[it]["id"], ds=items[it]["ds"], gen_by=items[it]["gen_by"], loop=bool(loopf[i]), pos=int(posn[i]),
                pre=tok.decode(TOK[s:i]), tok=tok.decode(TOK[i:i + 1]), post=tok.decode(TOK[i + 1:e]))
SCAN = 6_000_000
for nm, (L, sae) in SAES.items():
    sel = torch.tensor(sel_all[nm]).cuda()
    out = {}
    for split, idx0 in [("train", tr_idx), ("test", te_idx)]:
        idx = block_sample(~istest & ok if split == "train" else istest & ok, SCAN if split == "train" else 2_000_000)
        vals = []
        for s in range(0, len(idx), 16384):
            x = torch.from_numpy(np.asarray(A[L][idx[s:s + 16384]])).cuda().float()
            vals.append(sae.encode(x, sel).half().cpu())
        V = torch.cat(vals).float()   # [n, nsel]
        for c, j in enumerate(sel_all[nm]):
            v = V[:, c]; o = out.setdefault(j, {})
            if split == "train":
                order = torch.argsort(v, descending=True)[:3000].numpy(); ex, per_id, wins = [], {}, set()
                for q in order:
                    if v[q] <= 0: break
                    i = idx[q]; key = items[item[i]]["id"]
                    if per_id.get(key, 0) >= 2 or (item[i], posn[i] // 64) in wins: continue
                    per_id[key] = per_id.get(key, 0) + 1; wins.add((item[i], posn[i] // 64))
                    e = ctx(i); e["act"] = float(v[q]); ex.append(e)
                    if len(ex) >= 15: break
                o["top"] = ex; o["act_rate_train"] = float((v > 0).float().mean())
                o["ds_dist_top"] = {d: sum(e["ds"] == d for e in ex) for d in set(e["ds"] for e in ex)}
            else:
                pos_ = torch.nonzero(v > 0).flatten().numpy(); neg_ = torch.nonzero(v == 0).flatten().numpy()
                if len(pos_) >= 6:
                    srt = pos_[np.argsort(v[pos_].numpy())]; qs = [srt[int(len(srt) * f)] for f in [0.3, 0.5, 0.7, 0.85, 0.95, 0.999]]
                else: qs = list(pos_)
                negs = rng.choice(neg_, 6, replace=False)
                o["score_pos"] = [dict(ctx(idx[q]), act=float(v[q])) for q in qs]; o["score_neg"] = [ctx(idx[q]) for q in negs]
    R[nm]["exemplars"] = {int(j): o for j, o in out.items()}
    print("exemplars", nm, len(out), flush=True)
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT)
