"""Test 2 (CPU part): bare-direction reads of ±u15 / ±u19 with the logit lens and the public J-lens / J++ lens.
lens(v) = lm_head(final_norm(J_{L-1} v)) (scale-invariant up to eps). Also reads the 64 edited write vectors' Δ.
    python scripts/test2_bare_cpu.py OUT.json
"""
import sys, json, torch, jlens
from test24_common import *
m, tok = load_model("base", device="cpu"); _, layers, norm, head = lm_parts(m)
U = {k: torch.tensor(v) for k, v in get_u().items()}
CON, CNT = class_ids(tok, CONCLUDE), class_ids(tok, CONTINUE)
J = {"jlens_pub": jlens.JacobianLens.load(f"{DATA}/ext/jlens/Qwen3.5-0.8B_jacobian_lens.pt"),
     "jpp_pub": jlens.JacobianLens.load(f"{DATA}/ext/jpp/Qwen3.5-0.8B_jpp_lens.pt")}
def lens(nm, v, L):
    if nm != "logit" and L < 24: v = v @ J[nm].jacobians[L - 1].T
    with torch.no_grad(): return head(norm((v * 30)[None].to(torch.bfloat16)))[0].float()
out = {}
for nm in ["logit", "jlens_pub", "jpp_pub"]:
    for L in list(range(16, 24)) + [24]:
        for uk in [15, 19]:
            if uk == 19 and L < 20: continue
            for sg in [1, -1]:
                lg = lens(nm, sg * U[uk], L); lg = lg - lg.mean()
                out[f"{nm}|{L}|u{uk}|{'+' if sg > 0 else '-'}"] = dict(top=[tok.decode([t]) for t in lg.topk(20).indices.tolist()],
                    cls=float(lg[CON].mean() - lg[CNT].mean()))
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    if "|u15|" in k or "|u19|" in k: print(k, f"cls={v['cls']:+.2f}", " ".join(repr(t) for t in v["top"][:12]))
