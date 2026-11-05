"""Test 2: fit a tuned lens (Belrose et al. 2023) on base Qwen3.5-0.8B for block outputs 11..22 (hidden_states 12..23).

Translator per layer: lens_l(h) = lm_head(final_norm(h + A_l h + b_l)), A_l, b_l init 0 (= logit lens), trained with
KL(p_model || p_lens) at sampled positions of ~1000 contexts (test24_common.lens_contexts, train split). 10 % of contexts held out.
    python scripts/test2_fit_tuned.py OUT.pt
"""
import sys, time, json, numpy as np, torch
from test24_common import *
OUT = sys.argv[1]
LAY = list(range(12, 24))  # hidden_states indices
m, tok = load_model("base"); lm, layers, norm, head = lm_parts(m)
ctx = lens_contexts(tok, "train")
print("contexts", len(ctx), {k: sum(c[2] == k for c in ctx) for k in ["loop", "math", "nonmath", "wiki"]}, flush=True)
H = {L: [] for L in LAY}; F = []; kinds = []; ci = []
with torch.no_grad():
    for k, (ids, pos, kind, lp) in enumerate(ctx):
        cap = {}
        hk = layers[23].register_forward_hook(lambda mod, inp, out: cap.__setitem__("f", (out[0] if isinstance(out, tuple) else out)[0]))
        o = m.model(torch.tensor(ids[None].astype(np.int64)).cuda(), output_hidden_states=True); hk.remove()
        p = torch.tensor(pos).cuda()
        for L in LAY: H[L].append(o.hidden_states[L][0, p].half().cpu())
        F.append(cap["f"][p].half().cpu())   # output of block 23 = final residual BEFORE the final norm
        kinds += [kind] * len(pos); ci += [k] * len(pos)
        if k % 100 == 0: print(k, flush=True)
H = {L: torch.cat(v) for L, v in H.items()}; F = torch.cat(F); ci = np.array(ci)
te = torch.from_numpy(ci % 10 == 0); tr = ~te
print("positions", len(F), "train", int(tr.sum()), flush=True)
def logp(h): return torch.log_softmax(head(norm(h.to(torch.bfloat16))).float(), -1)
res, state = {}, {}
BS = 256
for L in LAY:
    A = torch.zeros(1024, 1024, device="cuda", requires_grad=True); b = torch.zeros(1024, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([A, b], lr=1e-3)
    idx = torch.nonzero(tr).flatten(); nstep = 2 * len(idx) // BS
    for st in range(nstep):
        j = idx[torch.randint(len(idx), (BS,))]
        h = H[L][j].cuda().float(); f = F[j].cuda()
        with torch.no_grad(): tgt = logp(f)
        lp = torch.log_softmax(head(norm((h + h @ A.T + b).to(torch.bfloat16))).float(), -1)
        loss = (tgt.exp() * (tgt - lp)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups: g["lr"] = 1e-3 * (1 - st / nstep)
    with torch.no_grad():
        jt = torch.nonzero(te).flatten(); kl_t = kl_l = 0
        for s in range(0, len(jt), BS):
            j = jt[s:s + BS]; h = H[L][j].cuda().float(); tgt = logp(F[j].cuda())
            lpt = logp(h + h @ A.T + b); lpl = logp(h)
            kl_t += (tgt.exp() * (tgt - lpt)).sum().item(); kl_l += (tgt.exp() * (tgt - lpl)).sum().item()
    res[L] = dict(kl_tuned=kl_t / len(jt), kl_logit=kl_l / len(jt))
    state[L] = (A.detach().cpu(), b.detach().cpu())
    print(L, res[L], flush=True)
torch.save(dict(translators=state, heldout=res, layers=LAY, n_pos=len(F), n_ctx=len(ctx)), OUT)
