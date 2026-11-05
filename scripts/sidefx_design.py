"""Design numbers for the side-effect variants (CPU, from the test-3 teacher-forced base collection; no new GPU).
Per Astra neuron: mean signed write along u, c_j = act_j * u^T w_j, at loop positions (loop ctx) vs normal thinking-on finished
positions. Per position: base resid.u after L15 / L19 and the Astra-64 write, to simulate the p50c push and a floor clamp.
    python scripts/sidefx_design.py <t3_collect_32768.pt> <out.json>"""
import sys, json, torch, numpy as np
D = torch.load(sys.argv[1], weights_only=False)
meta, S, st = D["meta"], D["series"], D["stats"]
cols = {L: np.asarray(D["cols"][L]).astype(int) for L in (15, 19)}
loop_i = [i for i, m in enumerate(meta) if m["cls"] == "loop"]
norm_i = [i for i, m in enumerate(meta) if m["cls"] == "normal" and m["kind"] == "think_finished"]
out = dict(n_loop_ctx=len(loop_i), n_normal_think_ctx=len(norm_i), layers={})
for L in (15, 19):
    s = st[L]
    nl = sum(meta[i]["n_loop"] for i in loop_i)
    nn = sum(meta[i]["n_loop"] + meta[i]["n_non"] for i in norm_i)
    mL = s["sum_loop"][loop_i].sum(0).numpy() / nl
    mN = (s["sum_loop"][norm_i].sum(0) + s["sum_non"][norm_i].sum(0)).numpy() / nn
    c = cols[L]
    # per-position series
    key_r, key_w = f"resid_u_{L + 1}", f"write_astra_{L}"
    rl, wl, rn, wn = [], [], [], []
    for i in loop_i:
        P = meta[i]["P"]; m = S[i]["m"].copy(); m[:P] = False
        rl.append(S[i][key_r].float().numpy()[m]); wl.append(S[i][key_w].float().numpy()[m])
    for i in norm_i:
        P = meta[i]["P"]; g = np.arange(meta[i]["T"]) >= P
        rn.append(S[i][key_r].float().numpy()[g]); wn.append(S[i][key_w].float().numpy()[g])
    rl, wl, rn, wn = map(np.concatenate, (rl, wl, rn, wn))
    pct = {p: float(np.percentile(rn, p)) for p in (1, 5, 10, 25, 50)}
    R = dict(mean_c_loop_astra=float(mL[c].sum()), mean_c_normal_astra=float(mN[c].sum()),
             resid_u_normal_mean=float(rn.mean()), resid_u_normal_pct=pct, resid_u_loop_mean=float(rl.mean()),
             write_loop_mean=float(wl.mean()), write_normal_mean=float(wn.mean()),
             per_neuron={int(j): dict(loop=float(mL[j]), normal=float(mN[j])) for j in c})
    for a in (2, 3, 4, 6):
        el, en = rl - a * wl, rn - a * wn
        R[f"alpha{a}"] = dict(edited_loop_mean=float(el.mean()), edited_normal_mean=float(en.mean()),
                              frac_loop_below_p5=float((el < pct[5]).mean()), frac_normal_below_p5=float((en < pct[5]).mean()))
    for fname, F in (("mean", float(rn.mean())), ("p5", pct[5]), ("p10", pct[10])):
        # clamp rule (as in the hook): new = max(edited, min(base, F)) ; push retained = base - new
        for which, r, w in (("loop", rl, wl), ("normal", rn, wn)):
            e = r - 6 * w; new = np.maximum(e, np.minimum(r, F))
            R[f"clamp6_{fname}_{which}"] = dict(floor=F, push_full=float((r - e).mean()), push_kept=float((r - new).mean()),
                                               new_mean=float(new.mean()), frac_bound=float((new > e + 1e-6).mean()))
    out["layers"][L] = R
json.dump(out, open(sys.argv[2], "w"), indent=1)
for L, R in out["layers"].items():
    print(L, {k: v for k, v in R.items() if k != "per_neuron"})
