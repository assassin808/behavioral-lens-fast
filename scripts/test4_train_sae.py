"""Test 4: train a BatchTopK SAE (Bussmann et al. 2024) on collected base activations.

    python scripts/test4_train_sae.py ACTDIR LAYER OUT.pt [D_SAE=16384] [K=64] [EPOCHS=1]

- input scaling: x / c with c = sqrt(E||x - mean||^2 / d) fixed from data (stored); positions with norm > 8x median
  (attention-sink positions) and prompt-free first 4 positions are excluded from training.
- train split only (items.json split == 'train'); held-out FVU / L0 / dead fraction reported on split == 'test'.
- inference threshold theta = EMA of the smallest selected activation (BatchTopK -> JumpReLU-style per-token at eval).
- aux-k loss (k_aux = 512, coef 1/32) on latents dead for > 5M tokens.
"""
import sys, json, time, numpy as np, torch, torch.nn as nn

ACT, LAYER, OUTP = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D_SAE = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
K = int(sys.argv[5]) if len(sys.argv) > 5 else 64
EPOCHS = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
BS, LR, D = 4096, 2e-4, 1024
torch.manual_seed(0); np.random.seed(0)

X = np.load(f"{ACT}/acts_L{LAYER}.npy", mmap_mode="r")
meta = np.load(f"{ACT}/meta.npz"); items = json.load(open(f"{ACT}/items.json"))
is_test = np.array([it["split"] == "test" for it in items])[meta["item"]]
print("loading", X.shape, flush=True)
t0 = time.time(); Xall = torch.from_numpy(np.ascontiguousarray(X)); print("loaded", time.time() - t0, flush=True)
nrm = Xall[::97].float().norm(dim=1); med = nrm.median()
good = torch.ones(len(Xall), dtype=torch.bool)
for s in range(0, len(Xall), 1 << 22):
    good[s:s + (1 << 22)] = Xall[s:s + (1 << 22)].float().norm(dim=1) < 8 * med
good &= torch.from_numpy(meta["pos"] >= 4)
tr_idx = torch.nonzero(good & ~torch.from_numpy(is_test)).flatten(); te_idx = torch.nonzero(good & torch.from_numpy(is_test)).flatten()
print("train", len(tr_idx), "test", len(te_idx), "excluded", int((~good).sum()), flush=True)
samp = Xall[tr_idx[torch.randperm(len(tr_idx))[:200000]]].float()
mu = samp.mean(0); c = ((samp - mu).pow(2).sum(1).mean() / D).sqrt()
print("scale c", float(c), "median norm", float(med))


class SAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.b_dec = nn.Parameter(mu.clone() / c)
        W = torch.randn(D_SAE, D); W = W / W.norm(dim=1, keepdim=True)
        s.W_dec = nn.Parameter(W.clone()); s.W_enc = nn.Parameter(W.T.clone()); s.b_enc = nn.Parameter(torch.zeros(D_SAE))
        s.register_buffer("theta", torch.tensor(0.0))

    def pre(s, x): return torch.relu((x - s.b_dec) @ s.W_enc + s.b_enc)

    def topk(s, a, k):
        flat = a.flatten(); v, i = flat.topk(k * a.shape[0])
        z = torch.zeros_like(flat); z[i] = v; return z.view_as(a), v.min()

    def encode(s, x):  # inference (per-token threshold)
        a = s.pre(x); return a * (a > s.theta)

    def decode(s, z): return z @ s.W_dec + s.b_dec


sae = SAE().cuda(); opt = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
steps = int(EPOCHS * len(tr_idx) / BS); last_fire = torch.zeros(D_SAE, device="cuda"); tokseen = 0
perm = tr_idx[torch.randperm(len(tr_idx))]
print("steps", steps, flush=True)
for st in range(steps):
    j = perm[(st * BS) % len(perm):(st * BS) % len(perm) + BS]
    if len(j) < BS: continue
    x = Xall[j].cuda().float() / c
    a = sae.pre(x); z, thr = sae.topk(a, K)
    xh = sae.decode(z); err = x - xh
    l2 = err.pow(2).sum(1).mean()
    fired = (z > 0).any(0); tokseen += BS; last_fire[fired] = tokseen
    dead = (tokseen - last_fire) > 5_000_000
    aux = torch.tensor(0.0, device="cuda")
    if dead.sum() > 0:
        ad = a * dead[None]
        kk = min(512, int(dead.sum()))
        v, i = ad.topk(kk, dim=1); za = torch.zeros_like(ad).scatter_(1, i, v)
        eh = za @ sae.W_dec
        aux = (err.detach() - eh).pow(2).sum(1).mean() / 32
    loss = l2 + aux
    opt.zero_grad(); loss.backward()
    with torch.no_grad():  # remove decoder-norm-parallel gradient, keep unit norm
        Wd = sae.W_dec; Wd.grad -= (Wd.grad * Wd).sum(1, keepdim=True) * Wd
    lr = LR * min(1.0, (steps - st) / (0.2 * steps)); [g.__setitem__("lr", lr) for g in opt.param_groups]
    opt.step()
    with torch.no_grad():
        sae.W_dec /= sae.W_dec.norm(dim=1, keepdim=True)
        sae.theta.mul_(0.999).add_(0.001 * thr) if st > 0 else sae.theta.fill_(thr)
    if st % 500 == 0:
        fvu = (err.pow(2).sum(1).mean() / (x - x.mean(0)).pow(2).sum(1).mean()).item()
        print(f"step {st}/{steps} fvu {fvu:.3f} aux {aux.item():.4f} dead {int(dead.sum())} theta {sae.theta.item():.3f}", flush=True)

# held-out eval with threshold encoder
with torch.no_grad():
    fv_num = fv_den = 0; l0 = []; cnt = torch.zeros(D_SAE, device="cuda")
    sel = te_idx[torch.randperm(len(te_idx))[:400000]]
    xm = Xall[sel[:50000]].cuda().float().mean(0) / c
    for s in range(0, len(sel), BS):
        x = Xall[sel[s:s + BS]].cuda().float() / c
        z = sae.encode(x); xh = sae.decode(z)
        fv_num += (x - xh).pow(2).sum().item(); fv_den += (x - xm).pow(2).sum().item()
        l0.append((z > 0).sum(1).float().mean().item()); cnt += (z > 0).sum(0)
    res = dict(layer=LAYER, d_sae=D_SAE, k=K, fvu=fv_num / fv_den, l0=float(np.mean(l0)), dead_frac=float((cnt == 0).float().mean()),
               n_train_tokens=int(len(tr_idx)), steps=steps, n_eval=len(sel), scale=float(c))
print(json.dumps(res))
torch.save(dict(state=sae.state_dict(), scale=float(c), cfg=res), OUTP)
