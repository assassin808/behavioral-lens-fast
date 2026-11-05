"""Write the edit specs (hook form) for the side-effect arms from notes/sidefx_design.json (pre-registered design).
    python scripts/sidefx_specs.py notes/sidefx_design.json notes/sidefx/specs"""
import sys, json, os, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
cols = {L: [int(c) for c in np.load(f"{REPO}/shared/astra_cols_L{L}.npy")] for L in (15, 19)}
def spec(alpha, floor=None, sel=None):
    s = {"layers": {}}
    for L in (15, 19):
        c = cols[L] if sel is None else sel[L]["cols"]
        a = alpha if sel is None else sel[L]["alpha"]
        s["layers"][str(L)] = dict(cols=c, alpha=a, floor=None if floor is None else floor[L])
    return s
fl_mean = {L: d["layers"][str(L)]["resid_u_normal_mean"] for L in (15, 19)}
fl_p5 = {L: d["layers"][str(L)]["resid_u_normal_pct"]["5"] for L in (15, 19)}
# loop-selective subset: neurons with loop_mean - 5 * normal_mean > 0; alpha raised so the mean loop-position push equals p50c's
sel, info = {}, {}
for L in (15, 19):
    pn = d["layers"][str(L)]["per_neuron"]
    l = {int(j): v["loop"] for j, v in pn.items()}; n = {int(j): v["normal"] for j, v in pn.items()}
    S = [j for j in cols[L] if l[j] - 5 * n[j] > 0]
    aS = 6 * sum(l.values()) / sum(l[j] for j in S)
    sel[L] = dict(cols=S, alpha=aS)
    info[L] = dict(n=len(S), alpha=aS, loop_push=aS * sum(l[j] for j in S), loop_push_p50c=6 * sum(l.values()),
                   normal_push=aS * sum(n[j] for j in S), normal_push_p50c=6 * sum(n.values()))
specs = {"p50chook": spec(6.0), "a3": spec(3.0), "clampmean": spec(6.0, fl_mean), "clampp5": spec(6.0, fl_p5), "loopsel": spec(None, sel=sel)}
for k, v in specs.items(): json.dump(v, open(f"{out}/{k}.json", "w"))
print(json.dumps(dict(floors_mean=fl_mean, floors_p5=fl_p5, loopsel=info), indent=1))
