"""Test 3a (CPU): per-neuron loop-vs-normal difference of the signed write along u, c_j = act_j * (u^T w_j), from
scripts/test3_collect.py; true top-64 per layer vs Astra's 64 columns; split-half noise ceiling; per-segment means of Astra's
neurons (math / IFEval / MMLU-Pro, loop / non-loop).
    python scripts/test3_analyze.py [MAXT] [out_json]       default out: notes/test3a_ranking_{MAXT}.json
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test3_common import REPO, DATA_OUT

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/notes/test3a_ranking_{MAXT}.json"
D = torch.load(f"{DATA_OUT}/t3_collect_{MAXT}{os.environ.get('SUFFIX', '')}.pt", weights_only=False)
meta = D["meta"]; cls = np.array([m["cls"] for m in meta]); src = np.array([m["src"] for m in meta])
nl = np.array([m["n_loop"] for m in meta], float); nn = np.array([m["n_non"] for m in meta], float)
res = dict(MAXT=MAXT, n_ctx={c: int((cls == c).sum()) for c in ["loop", "normal", "nonmath"]},
           mean_len={c: float(np.mean([m["T"] for m in meta if m["cls"] == c])) for c in ["loop", "normal", "nonmath"]},
           loop_positions_in_loop_ctx=int(nl[cls == "loop"].sum()), positions_in_normal_ctx=int(nn[cls == "normal"].sum()),
           normal_kinds={k: int(sum(m.get("kind") == k for m in meta)) for k in ["think_finished", "nothink_finished"]}, layers={})


def rank_of(score, cols):
    order = np.argsort(-score); r = np.empty_like(order); r[order] = np.arange(len(order)); return r[cols]


for L in [15, 19]:
    S = {k: v.double().numpy() for k, v in D["stats"][L].items()}
    cols = np.asarray(D["cols"][L]); P = D["P"][L].double().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        m_loop = S["sum_loop"] / nl[:, None]; m_non = S["sum_non"] / nn[:, None]
        a_loop = S["abs_loop"] / nl[:, None]; a_non = S["abs_non"] / nn[:, None]
    il = (cls == "loop") & (nl > 50); inn = (cls == "normal") & (nn > 50)

    def delta(il, inn):
        x, y = m_loop[il], m_non[inn]
        d = x.mean(0) - y.mean(0); t = d / np.sqrt(x.var(0, ddof=1) / len(x) + y.var(0, ddof=1) / len(y) + 1e-12)
        return d, t
    d, t = delta(il, inn)
    d_within = np.nanmean(m_loop[il] - m_non[il], 0)                         # loop vs non-loop positions, same loop texts
    da = a_loop[il].mean(0) - a_non[inn].mean(0)                                # |act| difference (no u weighting)
    top = np.argsort(-d)[:64]; top_t = np.argsort(-t)[:64]; top_w = np.argsort(-d_within)[:64]
    # split-half noise ceiling: top-64 overlap between two random halves of the contexts
    rng = np.random.default_rng(0); sh = []
    li, ni = np.where(il)[0], np.where(inn)[0]
    for _ in range(50):
        a, b = rng.permutation(li), rng.permutation(ni)
        h1 = np.zeros(len(cls), bool); h1[a[: len(a) // 2]] = True; h1[b[: len(b) // 2]] = True
        h2 = np.zeros(len(cls), bool); h2[a[len(a) // 2:]] = True; h2[b[len(b) // 2:]] = True
        t1 = set(np.argsort(-delta(il & h1, inn & h1)[0])[:64]); t2 = set(np.argsort(-delta(il & h2, inn & h2)[0])[:64])
        sh.append(len(t1 & t2))
    # astra-col overlap expected by chance for a uniform draw: 64*64/3584 = 1.14
    prev = {k: set(np.load(f"{REPO}/shared/top64_{k}_L{L}.npy").astype(int).tolist()) for k in ["by_dprime", "by_uproj", "combined"]}
    r = rank_of(d, cols)
    # non-math: within IFEval / MMLU-Pro texts, loop vs non-loop positions
    nm = {}
    for s in ["ifeval", "mmlupro"]:
        k = (src == s) & (nl > 50) & (nn > 50)
        nm[s] = dict(n_ctx=int(k.sum()), astra_mean_within_diff=float(np.nanmean((m_loop[k] - m_non[k])[:, cols].sum(1))) if k.any() else None,
                     astra_mean_nonloop=float(np.nanmean(m_non[(src == s) & (nn > 50)][:, cols].sum(1))),
                     astra_top64_overlap_with_true_top64_by_within_diff=int(len(set(np.argsort(-np.nanmean(m_loop[k] - m_non[k], 0))[:64]) & set(cols))) if k.any() else None)
    res["layers"][L] = dict(
        n_loop_ctx=int(il.sum()), n_normal_ctx=int(inn.sum()),
        overlap_astra_true_top64=int(len(set(top) & set(cols))), overlap_astra_top64_by_t=int(len(set(top_t) & set(cols))),
        overlap_astra_top64_within_loop_texts=int(len(set(top_w) & set(cols))),
        overlap_astra_top64_by_absact=int(len(set(np.argsort(-da)[:64]) & set(cols))),
        split_half_overlap_mean=float(np.mean(sh)), split_half_overlap_min=int(np.min(sh)), chance_overlap=64 * 64 / 3584,
        overlap_true_top64_with_xuanqi={k: len(set(top) & v) for k, v in prev.items()},
        astra_rank_median=float(np.median(r)), astra_n_in_top128=int((r < 128).sum()), astra_n_in_top256=int((r < 256).sum()),
        astra_n_negative_delta=int((d[cols] < 0).sum()),
        sum_delta_astra=float(d[cols].sum()), sum_delta_true_top64=float(d[top].sum()), sum_delta_all=float(d.sum()),
        share_astra_over_true_top64=float(d[cols].sum() / d[top].sum()),
        sum_delta_positive_all=float(d[d > 0].sum()),
        mean_write_u_astra_loop=float(m_loop[il][:, cols].sum(1).mean()), mean_write_u_astra_normal=float(m_non[inn][:, cols].sum(1).mean()),
        mean_write_u_top64_loop=float(m_loop[il][:, top].sum(1).mean()), mean_write_u_top64_normal=float(m_non[inn][:, top].sum(1).mean()),
        mean_write_u_all_loop=float(m_loop[il].sum(1).mean()), mean_write_u_all_normal=float(m_non[inn].sum(1).mean()),
        uTw_astra_mean=float(P[cols].mean()), uTw_top64_mean=float(P[top].mean()), uTw_all_absmean=float(np.abs(P).mean()),
        frac_true_top64_below_index_1792=float((top < 1792).mean()), true_top64=top.tolist(),
        astra_delta=dict(zip(map(int, cols), map(float, d[cols]))), nonmath=nm)
    print(L, json.dumps({k: v for k, v in res["layers"][L].items() if k not in ("true_top64", "astra_delta")}, indent=0), flush=True)
    np.save(f"{DATA_OUT}/test3_delta_L{L}_{MAXT}.npy", d)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
