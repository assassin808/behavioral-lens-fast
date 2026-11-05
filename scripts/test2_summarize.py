"""Test 2: tables (markdown to stdout) and figures from test2_read.py output.
    python scripts/test2_summarize.py READ.json TUNED.pt FIGPREFIX
"""
import sys, json, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); TUN = torch.load(sys.argv[2]); FP = sys.argv[3]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#52514e"]
LENSES = ["logit", "tuned", "jlens_pub", "jlens_ours", "jpp_pub"]; RL = list(range(16, 24))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.6})
print("## meta", json.dumps(R["meta"]), "\ns =", R["s"], "s/|h| =", R["s_rel_norm"])
print("\n### Lens fidelity: held-out KL(model || lens) [nats], tuned-lens fit")
print("| hs | logit | tuned |\n|---|---|---|")
for L, v in TUN["heldout"].items(): print(f"| {L} | {v['kl_logit']:.2f} | {v['kl_tuned']:.2f} |")

def tok(lst, n=10): return " ".join(repr(t)[1:-1] for t in lst[:n])
print("\n### Bare-direction reads of u15 (top 10 tokens), per lens")
for nm in LENSES:
    print(f"\n**{nm}**\n\n| hs | +u top | −u top | cls(+u) |\n|---|---|---|---|")
    for L in RL + [24]:
        a, b = R["bare"].get(f"{nm}|{L}|u15|+"), R["bare"].get(f"{nm}|{L}|u15|-")
        if a: print(f"| {L} | `{tok(a['top'])}` | `{tok(b['top'])}` | {a['cls']:+.2f} |")
print("\n### Base-relative reads at real loop states (h + s·(±u15)), top-10 by mean Δp, and CONCLUDE−CONTINUE Δlogp score")
for nm in LENSES + ["rlens"]:
    print(f"\n**{nm}**\n\n| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |\n|---|---|---|---|---|---|---|")
    for L in RL:
        pre = f"rlens|{L}" if nm == "rlens" else f"rel|{nm}|{L}"
        g = lambda sk, sg, kind: R["reads"].get(f"{pre}|{sk}|{sg}|{kind}")
        if g("loop", -1, "dp") is None: continue
        print(f"| {L} | `{tok(g('loop', -1, 'dp')['up'])}` | `{tok(g('loop', 1, 'dp')['up'])}` | {g('loop', -1, 'dlogp')['cls']:+.3f} | {g('loop', 1, 'dlogp')['cls']:+.3f} | "
              f"{g('prodpos', -1, 'dlogp')['cls']:+.3f} | {g('prodpos', 1, 'dlogp')['cls']:+.3f} |")
# P2a tally
print("\n### P2a tally: layers (of 8) where cls(−u) > cls(+u), loop positions")
p2a = {}
for nm in LENSES + ["rlens"]:
    n = 0
    for L in RL:
        pre = f"rlens|{L}" if nm == "rlens" else f"rel|{nm}|{L}"
        n += R["reads"][f"{pre}|loop|-1|dlogp"]["cls"] > R["reads"][f"{pre}|loop|1|dlogp"]["cls"]
    nb = sum(R["bare"][f"{nm}|{L}|u15|-"]["cls"] > R["bare"][f"{nm}|{L}|u15|+"]["cls"] for L in RL) if nm != "rlens" else None
    p2a[nm] = (n, nb); print(f"- {nm}: base-relative {n}/8; bare {nb}/8")
print("\n### Δresid (p50c − base) at loop positions")
print("| hs | rel ‖Δ‖ | cos(Δ,−u15) | corr(lens Δ, lens −u) logit / tuned / jlens_pub / jlens_ours / jpp | top-8 tokens up (tuned, Δp) | top-8 down (tuned, Δp) |\n|---|---|---|---|---|---|")
for L in RL + [24]:
    d = R["delta"][str(L)]; cc = " / ".join(f"{R['corr_dresid_vs_minus_u'].get(f'{nm}|{L}', float('nan')):+.2f}" for nm in LENSES)
    key = f"dresid|tuned|{L}|dp" if L < 24 else "dresid|logit|24|dp"
    r = R["reads"].get(key, {"up": [], "down": []})
    print(f"| {L} | {d['rel_norm']:.2f} | {d['cos_minus_u15']:+.2f} | {cc} | `{tok(r['up'], 8)}` | `{tok(r['down'], 8)}` |")
print("\nfinal Δlogp vs R-lens(−u at layer L) correlation:", {k.split('|')[1]: round(v, 2) for k, v in R["corr_dresid_vs_minus_u"].items() if k.startswith("final")})
print("next-token (loop continuation) Δlogp at loop positions:", round(R["nexttok_dlogp_loop"], 3))
print("\n### Variance share of u in the residual (base, generated positions of all analysis items)")
print("| hs | n | u15 share | rank of u15 among eigvals | u19 share | top-1 PC share | mean h·u15 loop / non-loop gen / wiki |\n|---|---|---|---|---|---|---|")
V = R["variance"]
for L in sorted(V, key=int):
    v = V[L]
    print(f"| {L} | {v['n']} | {v['u15_share']:.4f} ({v['u15_share'] * 1024:.1f}× iso) | {v['u15_rank_among_eigs']} | {v['u19_share']:.4f} | {v['top1_pc_share']:.3f} | "
          f"{v['u15_mean_loop']:.2f} / {v['u15_mean_prod']:.2f} / {v['u15_mean_wiki']:.2f} |")

# ---- figures
Ls = sorted(V, key=int); x = [int(l) for l in Ls]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(x, [V[l]["u15_share"] * 1024 for l in Ls], "-o", color=C[0], lw=2, ms=4, label="u15")
ax[0].plot(x, [V[l]["u19_share"] * 1024 for l in Ls], "-o", color=C[1], lw=2, ms=4, label="u19")
ax[0].axhline(1, color=C[5], lw=1, ls="--"); ax[0].text(x[0], 1.15, "isotropic", color=C[5], fontsize=8)
ax[0].axvline(16, color="#bbb", lw=0.8); ax[0].axvline(20, color="#bbb", lw=0.8)
ax[0].set_xlabel("hidden_states index (= after block L−1)"); ax[0].set_ylabel("var(h·u) / tr Cov(h)  × 1024"); ax[0].set_yscale("log"); ax[0].legend(frameon=False)
ax[0].set_title("u's share of residual variance (base)")
for k, (g, lab) in enumerate([("loop", "loop positions"), ("prod", "non-loop generated pos."), ("wiki", "WikiText")]):
    ax[1].plot(x, [V[l][f"u15_mean_{g}"] for l in Ls], "-o", color=C[k], lw=2, ms=4, label=lab)
ax[1].set_xlabel("hidden_states index"); ax[1].set_ylabel("mean h·u15"); ax[1].legend(frameon=False); ax[1].set_title("projection on u15 (base)")
fig.tight_layout(); fig.savefig(FP + "_variance.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for k, nm in enumerate(LENSES + ["rlens"]):
    pre = (lambda L: f"rlens|{L}") if nm == "rlens" else (lambda L, nm=nm: f"rel|{nm}|{L}")
    y = [R["reads"][f"{pre(L)}|loop|-1|dlogp"]["cls"] - R["reads"][f"{pre(L)}|loop|1|dlogp"]["cls"] for L in RL]
    ax[0].plot(RL, y, "-o", color=C[k], lw=2 if nm in ("tuned", "rlens", "jlens_ours") else 1, ms=4, label=nm)
ax[0].axhline(0, color="#888", lw=0.8); ax[0].set_xlabel("hidden_states index where ±s·u15 is added"); ax[0].set_ylabel("cls(−u) − cls(+u)  [Δlogp]")
ax[0].set_title("CONCLUDE−CONTINUE: −u vs +u (loop pos.)"); ax[0].legend(frameon=False, fontsize=7)
for k, nm in enumerate(LENSES):
    ax[1].plot(RL + [24], [R["corr_dresid_vs_minus_u"].get(f"{nm}|{L}", np.nan) for L in RL + [24]], "-o", color=C[k], lw=1.5, ms=4, label=nm)
ax[1].axhline(0, color="#888", lw=0.8); ax[1].set_xlabel("hidden_states index"); ax[1].set_ylabel("Pearson over vocab")
ax[1].set_title("lens(Δresid of edit) vs lens(−s·u15)"); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FP + "_axis_dresid.png", dpi=150)
print("\nfigures:", FP + "_variance.png", FP + "_axis_dresid.png")
