"""Figures for test 5 from notes/test5_paths_<TAG>.json.   python scripts/test5_figs.py TAG"""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TAG = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{REPO}/notes/test5_paths_{TAG}.json")); C = S["comp"]
os.makedirs(f"{REPO}/notes/fig", exist_ok=True)
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29d", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
sub = f"n={S['n_texts']} looping texts × {S['maxt']} tok, {S['n_loop_pos']} loop positions; Δ_total loop NLL = {S['total']:+.3f}"

# 1. mediators: layer blocks + groups, block vs inject (% of total), with noise / perm controls
names = [f"mlp{L}" for L in range(15, 24)] + [f"attn{L}" for L in range(16, 24)] + \
        ["attn16-23", "attn22-23", "fullattn19+23", "mlp16-18", "mlp20-23", "mlp22-23", "layers22-23", "all_downstream(excl mlp19)"]
names = [n for n in names if n in C]
y = np.arange(len(names))[::-1]
fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(names) + 1.4))
ax.barh(y + 0.2, [100 * C[n]["block_frac"] for n in names], 0.38, color=BLUE, label="block: p50c run, X ← base", edgecolor="white", linewidth=1)
ax.barh(y - 0.2, [100 * C[n]["inject_frac"] for n in names], 0.38, color=ORANGE, label="inject: base run, X ← p50c", edgecolor="white", linewidth=1)
ax.scatter([100 * C[n].get("ctrl_noise", np.nan) / S["total"] for n in names], y - 0.2, s=14, color=INK, marker="x", label="control: norm-matched noise", zorder=3)
ax.scatter([100 * C[n].get("ctrl_perm", np.nan) / S["total"] for n in names], y - 0.2, s=14, facecolor="none", edgecolor=INK, marker="o", label="control: position-permuted Δ", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(names); ax.axvline(0, color=GRAY, lw=0.8); ax.axvline(100, color=GRAY, lw=0.8, ls=":")
ax.set_xlabel("% of the edit's loop-NLL effect"); ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
ax.set_title("Which components carry the p50c edit to the loop logits\n" + sub, fontsize=9, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2)
fig.tight_layout(); fig.savefig(f"{REPO}/notes/fig/test5_mediators_{TAG}.png", dpi=160); plt.close(fig)

# 2. per-head block fraction heatmap (L16-23)
M = np.full((8, 16), np.nan)
for i, L in enumerate(range(16, 24)):
    for h in range(16):
        if f"h{L}.{h}" in C: M[i, h] = 100 * C[f"h{L}.{h}"]["block_frac"]
cm = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
v = np.nanmax(np.abs(M)); fig, ax = plt.subplots(figsize=(7.5, 3.4))
im = ax.imshow(M, cmap=cm, vmin=-v, vmax=v, aspect="auto")
for i in range(8):
    for h in range(16):
        if not np.isnan(M[i, h]) and abs(M[i, h]) >= 0.25 * v:
            ax.text(h, i, f"{M[i,h]:.0f}", ha="center", va="center", fontsize=7, color=INK)
ax.set_yticks(range(8)); ax.set_yticklabels([f"L{L}{' (full)' if L in (19, 23) else ''}" for L in range(16, 24)])
ax.set_xticks(range(16)); ax.set_xlabel("head (linear-attn layers: 16 heads × 128; full-attn: 8 × 256)")
fig.colorbar(im, ax=ax, label="block, % of total effect")
ax.set_title("Per-head mediation (block X ← base in the p50c run)\n" + sub, fontsize=9, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{REPO}/notes/fig/test5_heads_{TAG}.png", dpi=160); plt.close(fig)

# 3. AtP vs exact inject
sn = [n for n in C if "atp_mean" in C[n]]
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for key, col, lab in (("atp_base", BLUE, "AtP, gradient at base"), ("atp_edit", ORANGE, "AtP, gradient at p50c")):
    ax.scatter([C[n]["inject"] for n in sn], [C[n][key] for n in sn], s=12, color=col, label=lab, alpha=0.8, edgecolor="white", linewidth=0.5)
lim = max(abs(np.array([C[n][k] for n in sn for k in ("inject", "atp_base", "atp_edit")]))) * 1.05
ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=0.8, ls=":")
for n in sorted(sn, key=lambda n: -abs(C[n]["inject"]))[:6]:
    ax.annotate(n, (C[n]["inject"], C[n]["atp_base"]), fontsize=7, color=MUTED, xytext=(3, 3), textcoords="offset points")
r = S["atp_vs_inject_spearman"]
ax.set_xlabel("exact inject effect (Δ loop NLL)"); ax.set_ylabel("AtP estimate")
ax.set_title(f"AtP vs exact patching, Spearman ρ base {r['base']:.2f} / p50c {r['edit']:.2f}", fontsize=9, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{REPO}/notes/fig/test5_atp_{TAG}.png", dpi=160); plt.close(fig)
print("figs written for", TAG)
