"""Test 8 (CPU): numbers + figures from scripts/test8_tf.py and scripts/test3_collect.py.
    python scripts/test8_analyze.py [MAXT] [out_json]     default out: notes/test8_results_{MAXT}.json, figures notes/fig/test8_*.png
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import REPO, DATA_OUT, RUNTAG
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
SUF = os.environ.get("SUFFIX", "")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/notes/test8_results_{MAXT}{SUF}_{RUNTAG}.json"
FIG = f"{REPO}/notes/fig"; os.makedirs(FIG, exist_ok=True)
T8 = torch.load(f"{DATA_OUT}/t8_tf_{MAXT}{SUF}.pt", weights_only=False)
PS = T8["passes"]; res = dict(MAXT=MAXT, probe=T8["probe_report"], passes={})
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, np.float64)))
U = {L: np.load(f"{REPO}/shared/u_L{L}.npy") for L in [15, 19]}


def runs(m, P):
    """(start, length) of consecutive True runs in m[P:]."""
    x = np.concatenate([[False], m[P:], [False]]).astype(int); d = np.diff(x)
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0]; return [(int(a + P), int(b - a)) for a, b in zip(s, e)]


def onsets(m, P, pre=64):
    """loop-run starts preceded by >= pre non-loop generated tokens."""
    out = []
    for s, l in runs(m, P):
        if s - P >= pre and not m[s - pre:s].any(): out.append(s)
    return out


# ---------------- per pass: loop statistics and probe readings
for key, R in PS.items():
    gen = sum(r["T"] - r["P"] for r in R); rs = [l for r in R for _, l in runs(r["m32"], r["P"])]
    d = dict(n_texts=len(R), gen_tokens=int(gen), mean_len=float(np.mean([r["T"] for r in R])),
             loop_frac32=float(np.mean([r["m32"][r["P"]:].mean() for r in R])), loop_frac16=float(np.mean([r["m16"][r["P"]:].mean() for r in R])),
             texts_with_loop32=int(sum(r["m32"][r["P"]:].any() for r in R)),
             n_runs32=len(rs), runs32_per_1k_tokens=1000 * len(rs) / max(gen, 1), run32_len_median=float(np.median(rs)) if rs else None,
             run32_len_mean=float(np.mean(rs)) if rs else None, run32_len_p90=float(np.percentile(rs, 90)) if rs else None,
             n_onsets=int(sum(len(onsets(r["m32"], r["P"])) for r in R)))
    for L in [15, 16, 19, 20, 24]:
        lp = np.concatenate([r[f"logit_{L}"][r["P"]:][r["m32"][r["P"]:]] for r in R]).astype(float)
        npp = np.concatenate([r[f"logit_{L}"][r["P"]:][~r["m32"][r["P"]:]] for r in R]).astype(float)
        ul = np.concatenate([r[f"u_{L}"][r["P"]:][r["m32"][r["P"]:]] for r in R]).astype(float)
        un = np.concatenate([r[f"u_{L}"][r["P"]:][~r["m32"][r["P"]:]] for r in R]).astype(float)
        d[f"L{L}"] = dict(P_loop_at_loop=float(sig(lp).mean()) if len(lp) else None, frac_gt05_at_loop=float((lp > 0).mean()) if len(lp) else None,
                          P_loop_at_nonloop=float(sig(npp).mean()), frac_gt05_at_nonloop=float((npp > 0).mean()),
                          u_loop=float(ul.mean()) if len(ul) else None, u_nonloop=float(un.mean()),
                          u_ratio=float(ul.mean() / un.mean()) if len(ul) else None)
    # probe-defined loop episodes (>= 16 consecutive tokens with P(loop) > 0.5) vs verbatim loops
    for L in [15, 16, 20]:
        eps, ov = 0, 0; tot_ep_tokens = 0
        for r in R:
            z = r[f"logit_{L}"].astype(float) > 0
            for s, l in runs(z, r["P"]):
                if l >= 16: eps += 1; tot_ep_tokens += l; ov += r["m32"][s:s + l].any()
        d[f"L{L}"].update(episodes16=eps, episodes_per_1k=1000 * eps / max(gen, 1), episodes_with_verbatim_loop=int(ov),
                          episode_token_frac=tot_ep_tokens / max(gen, 1))
    res["passes"][key] = d
    print(key, json.dumps({k: v for k, v in d.items() if not k.startswith("L")}), flush=True)
    for L in [15, 16, 20, 24]: print("   ", L, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d[f"L{L}"].items()}))

# ---------------- onset-locked averages
W0, W1 = 256, 512
def locked(R, field, mtag="m32"):
    xs = []
    for r in R:
        for s in onsets(r[mtag], r["P"]):
            if s + W1 <= r["T"]: xs.append(r[field][s - W0:s + W1].astype(float))
    return np.array(xs)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
lock = {}
for key, col in [("base|base_loop", "C0"), ("p50c|base_loop", "C3"), ("p50c|p50c_own", "C1"), ("base|p50c_own", "C2"), ("base|base_nothink", "C4")]:
    if key not in PS: continue
    for k, (field, tr) in enumerate([("logit_15", sig), ("logit_20", sig), ("u_20", lambda x: x)]):
        X = locked(PS[key], field)
        if len(X) == 0: continue
        y = tr(X).mean(0); ax[k].plot(np.arange(-W0, W1), y, color=col, label=f"{key} (n={len(X)})")
        lock.setdefault(key, {})[field] = dict(n_onsets=len(X), pre=float(tr(X[:, :W0]).mean()), post_0_128=float(tr(X[:, W0:W0 + 128]).mean()),
                                               post_128_512=float(tr(X[:, W0 + 128:]).mean()))
for k, t in enumerate(["P(loop), hidden_states[15] (input of L15, pre-edit)", "P(loop), hidden_states[20] (after L19)", "resid·u19, hidden_states[20]"]):
    ax[k].axvline(0, color="k", lw=.5); ax[k].set_title(t, fontsize=9); ax[k].set_xlabel("tokens from verbatim-loop onset (32-gram)")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/test8_onset_locked_{RUNTAG}.png", dpi=130); plt.close()
res["onset_locked"] = lock

# ---------------- example time series: p50c own generations with the most loop runs, + base loop texts
def ts_fig(key, path, k=4):
    R = sorted(PS[key], key=lambda r: -len(runs(r["m32"], r["P"])))[:k]
    fig, ax = plt.subplots(k, 1, figsize=(14, 2.4 * k), squeeze=False)
    for a, r in zip(ax[:, 0], R):
        t = np.arange(r["T"]); a.fill_between(t, 0, 1, where=r["m32"], color="0.85", step="mid", label="verbatim loop (32-gram)")
        a.plot(t, sig(r["logit_15"].astype(float)), lw=.5, color="C0", label="P(loop) hs[15] pre-edit")
        a.plot(t, sig(r["logit_20"].astype(float)), lw=.5, color="C3", label="P(loop) hs[20]")
        a2 = a.twinx(); a2.plot(t, r["u_20"].astype(float), lw=.3, color="C2", alpha=.6); a2.set_ylabel("resid·u19 hs[20]", fontsize=7)
        a.set_xlim(r["P"], r["T"]); a.set_title(f"{key}  {r['id']}  closed={r.get('closed')}", fontsize=8)
    ax[0, 0].legend(fontsize=7, loc="upper left"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
for key, name in [("p50c|p50c_own", "p50c_own"), ("base|base_loop", "base_loop"), ("base|base_nothink", "base_nothink")]:
    if key in PS: ts_fig(key, f"{FIG}/test8_timeseries_{name}_{RUNTAG}.png")

# ---------------- thinking on vs off over the first N generated tokens (same problems, non-loop positions only)
if "base|first_think" in PS and "base|first_nothink" in PS:
    A = {r["id"]: r for r in PS["base|first_think"]}; B = {r["id"]: r for r in PS["base|first_nothink"]}
    ids = sorted(set(A) & set(B)); tv = {}
    for N in [256, 1024, 2048]:
        for L in [16, 20]:
            def mean_first(r):
                x = r[f"u_{L}"][r["P"]:r["P"] + N].astype(float); m = r["m32"][r["P"]:r["P"] + N]
                return x[~m].mean() if (~m).any() else np.nan
            a = np.array([mean_first(A[i]) for i in ids]); b = np.array([mean_first(B[i]) for i in ids]); ok = ~np.isnan(a) & ~np.isnan(b)
            bl = res["passes"]["base|base_loop"][f"L{L}"]; gap = bl["u_loop"] - bl["u_nonloop"]
            tv[f"N{N}_hs{L}"] = dict(n_problems=int(ok.sum()), think=float(a[ok].mean()), nothink=float(b[ok].mean()), diff=float((a - b)[ok].mean()),
                                     diff_se=float((a - b)[ok].std() / np.sqrt(ok.sum())), frac_think_higher=float(((a - b)[ok] > 0).mean()),
                                     loop_gap_base_think=float(gap), diff_over_loop_gap=float((a - b)[ok].mean() / gap))
    # also the loop gap within thinking-off loops
    res["think_vs_nothink"] = tv; print("think vs nothink", json.dumps(tv, indent=0))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, L in enumerate([16, 20]):
        for D_, lab, c in [(A, "thinking on", "C1"), (B, "thinking off", "C0")]:
            X = np.array([D_[i][f"u_{L}"][D_[i]["P"]:D_[i]["P"] + 2048].astype(float) for i in ids if D_[i]["T"] - D_[i]["P"] >= 2048])
            if len(X): ax[k].plot(np.convolve(X.mean(0), np.ones(32) / 32, "valid"), color=c, label=f"{lab} (n={len(X)})")
        bl = res["passes"]["base|base_loop"][f"L{L}"]
        ax[k].axhline(bl["u_loop"], ls="--", color="C3", lw=.8, label="base thinking-on loop positions"); ax[k].axhline(bl["u_nonloop"], ls=":", color="k", lw=.8, label="base thinking-on non-loop (loop texts)")
        ax[k].set_title(f"resid·u at hidden_states[{L}], first 2048 generated tokens (32-token smoothing)", fontsize=8); ax[k].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/test8_think_vs_nothink_u_{RUNTAG}.png", dpi=130); plt.close()

# ---------------- state geometry: PCA of base states (hs[20]) + projection on u
D3 = torch.load(f"{DATA_OUT}/t3_collect_{MAXT}{SUF}.pt", weights_only=False)
meta = D3["meta"]
Hn = np.concatenate([s["H"][20].float().numpy()[~s["is_loop"]] for s in D3["states"] if meta[s["ctx"]]["cls"] == "normal"])
def grab(key, which):
    X = []
    for r in PS.get(key, []):
        if "pca_idx" not in r: continue
        mk = r["m32"][r["pca_idx"]]; sel = mk if which == "loop" else ~mk
        X.append(r["pca_H20"][sel].astype(np.float32))
    return np.concatenate(X) if X else np.zeros((0, 1024), np.float32)
Hbl, Hel = grab("base|base_loop", "loop"), grab("p50c|base_loop", "loop")
Hpl, Hpn = grab("p50c|p50c_own", "loop"), grab("p50c|p50c_own", "non")
rng = np.random.default_rng(0)
sub = lambda X, n=3000: X[rng.choice(len(X), min(n, len(X)), replace=False)] if len(X) else X
Hn, Hbl, Hel, Hpl, Hpn = map(sub, [Hn, Hbl, Hel, Hpl, Hpn])
fitX = np.concatenate([Hn, Hbl]); mu = fitX.mean(0)
_, _, Vt = np.linalg.svd(fitX - mu, full_matrices=False); V = Vt[:2]
u = U[19] / np.linalg.norm(U[19])
Xo = (fitX - mu) - np.outer((fitX - mu) @ u, u); _, _, Vo = np.linalg.svd(Xo, full_matrices=False); vo = Vo[0]
cn = Hn.mean(0)
geo = dict(n_states=dict(base_normal=len(Hn), base_loop=len(Hbl), edited_on_base_loop=len(Hel), p50c_own_loop=len(Hpl), p50c_own_nonloop=len(Hpn)),
           cos_pc1_u=float(abs(V[0] @ u)), cos_pc2_u=float(abs(V[1] @ u)),
           dist_to_normal_centroid=dict(base_loop=float(np.linalg.norm(Hbl - cn, axis=1).mean()), edited_loop=float(np.linalg.norm(Hel - cn, axis=1).mean()) if len(Hel) else None,
                                        base_normal=float(np.linalg.norm(Hn - cn, axis=1).mean())),
           cos_displacement_edited_minus_base_loop_u=float(((Hel.mean(0) - Hbl.mean(0)) @ u) / np.linalg.norm(Hel.mean(0) - Hbl.mean(0))) if len(Hel) else None,
           centroid_u=dict(base_normal=float(Hn.mean(0) @ u), base_loop=float(Hbl.mean(0) @ u), edited_loop=float(Hel.mean(0) @ u) if len(Hel) else None,
                           p50c_own_loop=float(Hpl.mean(0) @ u) if len(Hpl) else None, p50c_own_nonloop=float(Hpn.mean(0) @ u) if len(Hpn) else None))
res["geometry_hs20"] = geo; print("geometry", json.dumps(geo, indent=0))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for X, lab, c in [(Hn, "base, normal texts", "0.6"), (Hbl, "base, loop positions", "C0"), (Hel, "p50c on the same base loop texts (edited loop states)", "C3"),
                  (Hpn, "p50c own gens, non-loop", "C2"), (Hpl, "p50c own gens, verbatim-loop positions", "C1")]:
    if not len(X): continue
    Z = (X - mu) @ V.T; ax[0].scatter(Z[:, 0], Z[:, 1], s=2, alpha=.35, color=c, label=f"{lab} (n={len(X)})")
    ax[1].scatter((X - mu) @ u, (X - mu) @ vo, s=2, alpha=.35, color=c, label=lab)
ax[0].set_title("PCA (fit on base normal + base loop states), hidden_states[20]", fontsize=9); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[1].set_title("x = resid·u19 (centred), y = top PC orthogonal to u", fontsize=9); ax[1].set_xlabel("u"); ax[1].set_ylabel("PC1 ⟂ u")
lg = ax[0].legend(fontsize=7, markerscale=4); plt.tight_layout(); plt.savefig(f"{FIG}/test8_state_pca_{RUNTAG}.png", dpi=130); plt.close()
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
