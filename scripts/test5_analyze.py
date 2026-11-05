"""Test 5 analysis: mediator table from a main run + replication from a second run's per-text .pt.
    python scripts/test5_analyze.py MAIN_TAG MAIN_PT REP_PT OUT_MD
Replication: (a) same texts on other hardware (per-text agreement), (b) texts in REP but not in MAIN (held-out)."""
import json, sys, os
import numpy as np, torch
MAIN, MPT, RPT, OUT = sys.argv[1:5]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{REPO}/notes/test5_paths_{MAIN}.json")); C = S["comp"]; t = S["total"]
RM, RR = torch.load(MPT, weights_only=False), torch.load(RPT, weights_only=False)
out = []
P = out.append

def per_text(R, key, n):
    b = np.array([r["base"][0] for r in R]); e = np.array([r["edit"][0] for r in R])
    if key == "total": return e - b
    if key == "inject": return np.array([r["inject"][n][0] for r in R]) - b
    if key == "block": return e - np.array([r["block"][n][0] for r in R])
def frac(R, key, n):
    return per_text(R, key, n).mean() / per_text(R, "total", None).mean()

ids_m = [r["id"] for r in RM]; heldout = [r for r in RR if r["id"] not in ids_m]; same = [r for r in RR if r["id"] in ids_m]
P(f"main: {MAIN}, n={len(RM)} texts x {RM[0]['T']} tok, {sum(r['n_loop'] for r in RM)} loop positions; total {t:+.4f} "
  f"(base {S['M_base']:.4f} -> p50c {S['M_edit']:.4f}); prod NLL {S['prod_base']:.3f} -> {S['prod_edit']:.3f}")
P(f"replication run: n={len(RR)}; {len(same)} texts shared with main, {len(heldout)} held-out texts "
  f"({sum(r['n_loop'] for r in heldout)} loop positions)")
if same:
    mm = {r["id"]: r for r in RM}
    d = [abs((r["edit"][0] - r["base"][0]) - (mm[r["id"]]["edit"][0] - mm[r["id"]]["base"][0])) for r in same]
    d2 = [abs((r["edit"][0] - r["block"]["attn16-23"][0]) - (mm[r["id"]]["edit"][0] - mm[r["id"]]["block"]["attn16-23"][0])) for r in same]
    P(f"cross-hardware, same texts: max |Δtotal diff| {max(d):.4f}, max |block attn16-23 diff| {max(d2):.4f}")
names = ["mlp15", "mlp19", "attn16-23", "attn16-21", "attn22-23", "fullattn19+23", "attn19", "attn23",
         "mlp16-23(excl19)", "mlp16-18", "mlp20-23", "mlp22-23", "mlp23", "layers22-23", "all_downstream(excl mlp19)",
         "h23.5", "h23.7", "h23.0", "h23.2", "h23.6", "h19.4", "h19.6", "h18.14", "h16.6", "h17.6", "h22.2", "h22.8"]
P("")
P("| component | block % (main) [95% CI] | inject % (main) | noise ctrl % | perm ctrl % | direct % | block % held-out | inject % held-out |")
P("|---|---|---|---|---|---|---|---|")
for n in names:
    c = C[n]; ci = [100 * x / t for x in c["block_ci"]]
    ho_b = 100 * frac(heldout, "block", n) if heldout else float("nan"); ho_i = 100 * frac(heldout, "inject", n) if heldout else float("nan")
    dr = f"{100*c['direct']/t:.0f}" if "direct" in c else "–"
    nz = f"{100*c['ctrl_noise']/t:.0f}" if "ctrl_noise" in c else "–"; pm = f"{100*c['ctrl_perm']/t:.0f}" if "ctrl_perm" in c else "–"
    P(f"| {n} | {100*c['block_frac']:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] | {100*c['inject_frac']:.0f} | {nz} | {pm} | {dr} | {ho_b:.0f} | {ho_i:.0f} |")
P("")
P(f"direct (last hop only): edit sites mlp15+mlp19 {100*S['direct_edit_sites']/t:.1f} %, all Δ {100*S['direct_ALL']/t:.1f} %")
P(f"edit L15 only {100*S['edit15only']/t:.0f} %, L19 only {100*S['edit19only']/t:.0f} %")
P(f"AtP vs exact inject Spearman: {S['atp_vs_inject_spearman']}")
for n in ["mlp15", "mlp19", "attn23", "mlp23", "attn19"]:
    c = C[n]; P(f"  AtP {n}: base-grad {c['atp_base']:+.4f}  p50c-grad {c['atp_edit']:+.4f}  exact inject {c['inject']:+.4f}  exact block {c['block']:+.4f}")
P("")
P("position split of inject (% total): all / at loop (query) positions only / at all other positions")
for n in [f"attn{L}" for L in (19, 22, 23)] + ["mlp15", "mlp19", "h23.5", "h23.7", "h19.4"]:
    c = C[n]; P(f"  {n}: {100*c['inject']/t:.1f} / {100*c['inject_looppos']/t:.1f} / {100*c['inject_nonlooppos']/t:.1f}")
P("")
P("induction attention (query p -> earlier position holding the repeated token), mean over loop positions, per head; base -> p50c")
for L in (15, 19, 23):
    b, e = S["copy"][f"base_L{L}"]["induction"], S["copy"][f"edit_L{L}"]["induction"]
    P(f"  L{L}: " + "  ".join(f"h{h} {x:.3f}->{y:.3f}" for h, (x, y) in enumerate(zip(b, e))) + f"   sum {sum(b):.3f}->{sum(e):.3f} ({100*(sum(e)/sum(b)-1):+.0f} %)")
    b, e = S["copy"][f"base_L{L}"]["duplicate"], S["copy"][f"edit_L{L}"]["duplicate"]
    P(f"    duplicate-token mass sum {sum(b):.3f}->{sum(e):.3f}")
P("")
P("logit lens of Δ_X (p50c − base output, linearised through the base final norm) at loop positions:")
P("| X | ‖Δ‖ | cos(Δ,u15) | consistency | centred Δlogit of repeated token | median percentile of target | top promoted | top suppressed |")
P("|---|---|---|---|---|---|---|---|")
for n in ["ALL", "mlp15", "mlp19", "attn19", "attn22", "attn23", "h23.5", "h23.7", "mlp22", "mlp23"]:
    l = S["lens"][n]
    P(f"| {n} | {l['dnorm']:.2f} | {l['cos_u15']:+.2f} | {l['consist']:.2f} | {l['target_centered']:+.2f} | {l['target_pct_median']:.3f} | "
      f"{', '.join(repr(a) for a, _ in l['promoted'][:6])} | {', '.join(repr(a) for a, _ in l['suppressed'][:6])} |")
P("")
P("Δlogit (centred) of discourse tokens, all-Δ / attn23 / mlp15: " + ", ".join(
    f"{k!r} {S['lens']['ALL']['stop'][k]:+.2f}/{S['lens']['attn23']['stop'][k]:+.2f}/{S['lens']['mlp15']['stop'][k]:+.2f}" for k in S["lens"]["ALL"]["stop"]))
P("")
h = S["h7"]
P(f"L15 h7: ablate in base Δloop {h['ablate_base']:+.4f} ({100*h['ablate_base']/t:.0f} % of p50c), Δprod {h['ablate_base_prod']:+.4f}; "
  f"control L15 h3 ablation {h['ctrl_h3_ablate_base']:+.4f}; extra when ablated on top of p50c {h['ablate_edit_extra']:+.4f}; "
  f"64 L15 neurons' u-write at loop positions {h['uwrite']['base_loop']:.3f} -> {h['uwrite']['abl_loop']:.3f} "
  f"({100*(h['uwrite']['abl_loop']/h['uwrite']['base_loop']-1):+.0f} %), prod {h['uwrite']['base_prod']:.3f} -> {h['uwrite']['abl_prod']:.3f}; "
  f"h7 output·u15 loop {h['out_u15_proj_loop']:+.3f} vs prod {h['out_u15_proj_prod']:+.3f} (cos {h['out_cos_u15_loop']:+.2f})")
if heldout:
    hb = np.array([r["h7_ablate_base"][0] - r["base"][0] for r in heldout]); he = np.array([r["h7_ablate_edit"][0] - r["edit"][0] for r in heldout])
    P(f"  held-out: h7 ablate base {hb.mean():+.4f}, extra on p50c {he.mean():+.4f}, total {per_text(heldout,'total',None).mean():+.4f}")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
