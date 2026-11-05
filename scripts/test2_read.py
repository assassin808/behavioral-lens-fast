"""Test 2: read u and the edit's Δresid with logit / tuned / J / J++ lenses and the R-lens (causal run-the-rest readout).

    python scripts/test2_read.py TUNED.pt JLENS_OURS.pt OUT.json [N_LOOP=40] [N_PROD=20] [N_WIKI=20] [MAXT=8192]

Layer convention: index L = hidden_states[L] = output of decoder block L-1 (residual, pre final norm; captured by hooks).
J-lens layer for index L is block L-1. Everything runs on the held-out (test) split of the corpus.
"""
import sys, json, time, numpy as np, torch, jlens
from test24_common import *
TUN, JOURS, OUT = sys.argv[1:4]
NL, NP, NW, MAXT = [int(a) for a in sys.argv[4:8]] if len(sys.argv) > 7 else (40, 20, 20, 8192)
LAYS = list(range(13, 25)); RL = list(range(16, 24))   # read u at 16..23 (outputs of blocks 15..22), Δresid also 24
POS_PER_ITEM = 200
torch.manual_seed(0); rng = np.random.default_rng(0)
base, tok = load_model("base"); edit, _ = load_model("p50c")
_, Lb, normb, headb = lm_parts(base); _, Le, _, _ = lm_parts(edit)
U = {k: torch.tensor(v).cuda() for k, v in get_u().items()}
CON, CNT = class_ids(tok, CONCLUDE), class_ids(tok, CONTINUE)
print("class tokens", [tok.decode([t]) for t in CON], [tok.decode([t]) for t in CNT], flush=True)

# ---------------- data ----------------
items = [x for x in chat_items(tok, maxt=MAXT) if x["split"] == "test"]
loopi = [x for x in items if x["gen_by"] == "unedited" and x["loop"].mean() > 0.03]
loopi = [loopi[i] for i in rng.permutation(len(loopi))[:NL]]
prodi = [x for x in items if x["ds"] in ("gsm8k", "math500") and x["finished"] and x["loop"].mean() < 0.01]
prodi = [prodi[i] for i in rng.permutation(len(prodi))[:NP]]
for x in prodi: x["ids"], x["loop"] = x["ids"][:4096], x["loop"][:4096]
wk = wiki_docs(1024 * (NW + 5), tok, seq=1024, skip_docs=8000)[:NW]
wikii = [dict(ids=w, P=16, loop=np.zeros(len(w), bool), id=f"wiki{i}") for i, w in enumerate(wk)]
print("n loop/prod/wiki items", len(loopi), len(prodi), len(wikii), "loop tok", sum(int(x["loop"].sum()) for x in loopi), flush=True)


def run(model, ids, add=None):
    """returns {L: [T,1024] fp32 residual} for L in LAYS, and final logits. add = (L, vec[T,1024]) added to output of block L-1."""
    layers = lm_parts(model)[1]; cap, hs = {}, []
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if add is not None and add[0] - 1 == i:
                o = o + add[1].to(o.dtype)
                cap[i + 1] = o[0].float()
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            cap[i + 1] = o[0].float()
        return h
    for i, l in enumerate(layers): hs.append(l.register_forward_hook(mk(i)))
    with torch.no_grad(): model.model(torch.tensor(ids[None].astype(np.int64)).cuda())
    for h in hs: h.remove()
    return {L: cap[L] for L in LAYS}, cap[24]   # final residual (pre-norm); logits = unemb(.) at chosen positions


# ---------------- lenses ----------------
tun = torch.load(TUN)["translators"]
Js = {"jlens_pub": jlens.JacobianLens.load(f"{DATA}/ext/jlens/Qwen3.5-0.8B_jacobian_lens.pt"),
      "jlens_ours": jlens.JacobianLens.load(JOURS),
      "jpp_pub": jlens.JacobianLens.load(f"{DATA}/ext/jpp/Qwen3.5-0.8B_jpp_lens.pt")}
Jc = {k: {l: v.jacobians[l].cuda() for l in v.source_layers} for k, v in Js.items()}
LENSES = ["logit", "tuned", "jlens_pub", "jlens_ours", "jpp_pub"]


def unemb(h): return headb(normb(h.to(torch.bfloat16))).float()


def lens(name, h, L, bare=False):
    if name == "logit" or L == 24: return unemb(h)
    if name == "tuned":
        A, b = tun[L]; A = A.cuda(); y = h + h @ A.T
        return unemb(y if bare else y + b.cuda())
    return unemb(h @ Jc[name][L - 1].T)


def cls_score(v):  # v: [V] mean Δlogp (or centered logit)
    return float(v[CON].mean() - v[CNT].mean())


def topk(v, k=15):
    return [tok.decode([t]) for t in v.topk(k).indices.tolist()], [tok.decode([t]) for t in (-v).topk(k).indices.tolist()]


R = {"meta": dict(n_loop_items=len(loopi), n_prod_items=len(prodi), n_wiki_items=len(wikii), maxt=MAXT, pos_per_item=POS_PER_ITEM,
                  conclude=[tok.decode([t]) for t in CON], cont=[tok.decode([t]) for t in CNT])}

# ---------------- 1. bare-direction reads ----------------
bare = {}
for nm in LENSES:
    for L in RL + [24]:
        for uk in [15, 19]:
            if uk == 19 and L < 20: continue
            for sg in [1, -1]:
                lg = lens(nm, (sg * U[uk])[None] * 30, L, bare=True)[0]; lg = lg - lg.mean()
                t, b = topk(lg)
                bare[f"{nm}|{L}|u{uk}|{'+' if sg > 0 else '-'}"] = dict(top=t, cls=cls_score(lg))
R["bare"] = bare
print("bare done", flush=True)

# ---------------- 2. variance share + Δresid + base-relative reads ----------------
acc = {}   # key -> sum vector over positions
cnt = {}
def add(key, v, n):
    acc[key] = acc.get(key, 0) + v; cnt[key] = cnt.get(key, 0) + n
var_stats = {L: dict(S=torch.zeros(1024, device="cuda", dtype=torch.float64), SS=torch.zeros(1024, 1024, device="cuda", dtype=torch.float64), n=0,
                     proj={k: {"loop": [], "prod": [], "wiki": []} for k in [15, 19]}) for L in LAYS}
dproj = {L: [] for L in LAYS}; drel = {L: [] for L in LAYS}; dcos = {L: [] for L in LAYS}; s_scale = []
nexttok = []
groups = [("loop", x) for x in loopi] + [("prod", x) for x in prodi] + [("wiki", x) for x in wikii]
t0 = time.time()
for gi, (g, x) in enumerate(groups):
    ids = x["ids"]; hb, lob = run(base, ids)
    gen = np.arange(max(x["P"], 16), len(ids) - 1)
    lp_ = x["loop"][gen]
    for L in LAYS:
        h = hb[L][gen].double(); st = var_stats[L]
        st["S"] += h.sum(0); st["SS"] += h.T @ h; st["n"] += len(gen)
        for k in [15, 19]:
            pr = (hb[L][gen] @ U[k]).cpu().numpy()
            if g == "loop": st["proj"][k]["loop"] += pr[lp_].tolist(); st["proj"][k]["prod"] += pr[~lp_].tolist()
            else: st["proj"][k][g] += pr.tolist()
    # positions to read
    if g == "loop":
        lpos = np.flatnonzero(x["loop"][:-1]); ppos = gen[~lp_]
        sets = {"loop": np.sort(rng.choice(lpos, min(POS_PER_ITEM, len(lpos)), replace=False)),
                "prodpos": np.sort(rng.choice(ppos, min(POS_PER_ITEM, len(ppos)), replace=False))}
    else:
        sets = {g: np.sort(rng.choice(gen, min(POS_PER_ITEM, len(gen)), replace=False))}
    if g == "loop":
        he, loe = run(edit, ids)
        p = torch.tensor(sets["loop"]).cuda()
        nxt = torch.tensor(ids[sets["loop"] + 1].astype(np.int64)).cuda()
        la, lb_ = torch.log_softmax(unemb(lob[p]), -1), torch.log_softmax(unemb(loe[p]), -1)
        nexttok.append(float((lb_.gather(1, nxt[:, None]) - la.gather(1, nxt[:, None])).mean()))
        for L in LAYS:
            d = he[L][p] - hb[L][p]
            dproj[L] += (d @ U[15]).tolist(); drel[L] += (d.norm(dim=1) / hb[L][p].norm(dim=1)).tolist()
            dcos[L] += torch.nn.functional.cosine_similarity(d, -U[15][None], dim=1).tolist()
            for nm in LENSES:
                if L < 16: continue
                a = torch.log_softmax(lens(nm, hb[L][p], L), -1); b_ = torch.log_softmax(lens(nm, he[L][p], L), -1)
                add(f"dresid|{nm}|{L}|dlogp", (b_ - a).sum(0), len(p)); add(f"dresid|{nm}|{L}|dp", (b_.exp() - a.exp()).sum(0), len(p))
        add("dresid|final|24|dlogp", (lb_ - la).sum(0), len(p)); del la, lb_
        del he
    if "s" not in R:   # s = median |Δ·u15| at loop positions at hidden_states[16], from the first loop item
        s = float(np.median(np.abs(dproj[16]))); R["s"] = s; R["s_rel_norm"] = float(np.median(np.abs(dproj[16])) / hb[16][sets['loop']].norm(dim=1).median().item())
        print("s", R["s"], "s/|h|", R["s_rel_norm"], flush=True)
    s = R["s"]
    # base-relative reads of ±u at real states
    for sk, pos in sets.items():
        p = torch.tensor(pos).cuda()
        for L in RL:
            for nm in LENSES:
                a = torch.log_softmax(lens(nm, hb[L][p], L), -1)
                for sg in [1, -1]:
                    b_ = torch.log_softmax(lens(nm, hb[L][p] + sg * s * U[15], L), -1)
                    add(f"rel|{nm}|{L}|{sk}|{sg}|dlogp", (b_ - a).sum(0), len(p)); add(f"rel|{nm}|{L}|{sk}|{sg}|dp", (b_.exp() - a.exp()).sum(0), len(p))
    # R-lens (causal): add ±s·u15 at all generated positions of block L-1's output; first 20 loop items + 10 prod + 10 wiki
    if (g == "loop" and gi < 20) or (g == "prod" and gi < len(loopi) + 10) or (g == "wiki" and gi < len(loopi) + len(prodi) + 10):
        T = len(ids)
        for L in RL:
            for sg in [1, -1]:
                vec = torch.zeros(1, T, 1024, device="cuda"); vec[0, max(x["P"], 16):] = sg * s * U[15]
                _, lo2 = run(base, ids, add=(L, vec))
                for sk, pos in sets.items():
                    p = torch.tensor(pos).cuda(); a0 = torch.log_softmax(unemb(lob[p]), -1); b2 = torch.log_softmax(unemb(lo2[p]), -1)
                    add(f"rlens|{L}|{sk}|{sg}|dlogp", (b2 - a0).sum(0), len(p)); add(f"rlens|{L}|{sk}|{sg}|dp", (b2.exp() - a0.exp()).sum(0), len(p))
    del hb
    print(gi, g, len(ids), f"{time.time() - t0:.0f}s", flush=True)

# ---------------- summarise ----------------
mean = {k: acc[k] / cnt[k] for k in acc}
R["n_pos"] = {k: cnt[k] for k in cnt}
R["reads"] = {}
for k, v in mean.items():
    if k.endswith("|dp"):
        t, b = topk(v, 20); R["reads"][k] = dict(up=t, down=b)
    else:
        t, b = topk(v, 20); R["reads"][k] = dict(up=t, down=b, cls=cls_score(v))
# correlation Δresid read vs −u base-relative read (same lens, same layer, loop positions)
R["corr_dresid_vs_minus_u"] = {}
for nm in LENSES:
    for L in RL + [24]:
        a = mean.get(f"dresid|{nm}|{L}|dlogp"); b = mean.get(f"rel|{nm}|{min(L, 23)}|loop|-1|dlogp")
        if a is None or b is None: continue
        R["corr_dresid_vs_minus_u"][f"{nm}|{L}"] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
for L in RL:
    a = mean.get("dresid|final|24|dlogp"); b = mean[f"rlens|{L}|loop|-1|dlogp"]
    R["corr_dresid_vs_minus_u"][f"final_dlogp_vs_rlens_minus_u|{L}"] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
R["nexttok_dlogp_loop"] = float(np.mean(nexttok))
R["delta"] = {L: dict(rel_norm=float(np.mean(drel[L])), cos_minus_u15=float(np.mean(dcos[L])), proj_u15_median=float(np.median(dproj[L]))) for L in LAYS}
vs = {}
for L in LAYS:
    st = var_stats[L]; mu = st["S"] / st["n"]; C = st["SS"] / st["n"] - torch.outer(mu, mu)
    ev = torch.linalg.eigvalsh(C).flip(0); tr = float(C.trace())
    row = dict(n=st["n"], top1_pc_share=float(ev[0] / tr), iso=1 / 1024)
    for k in [15, 19]:
        u = U[k].double(); vu = float(u @ C @ u)
        row[f"u{k}_share"] = vu / tr; row[f"u{k}_rank_among_eigs"] = int((ev > vu).sum())
        for g in ["loop", "prod", "wiki"]:
            row[f"u{k}_mean_{g}"] = float(np.mean(st["proj"][k][g])) if st["proj"][k][g] else None
    vs[L] = row
R["variance"] = vs
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT, time.time() - t0)
