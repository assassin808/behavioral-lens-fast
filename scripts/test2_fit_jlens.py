"""Test 2: fit our own J-lens (anthropics/jacobian-lens, unchanged estimator) on the loop / math / non-math mix.
Prompts = decoded 160-token windows (max_seq_len 128 as in the paper recipe): loop windows end inside a loop,
math / non-math windows from finished generations, WikiText windows.
    python scripts/test2_fit_jlens.py OUT.pt N_PROMPTS
"""
import os, sys, numpy as np, torch, jlens
from test24_common import *
OUT, N = sys.argv[1], int(sys.argv[2])
m, tok = load_model("base")
ctx = lens_contexts(tok, "train", seed=1)
rng = np.random.default_rng(1); prompts = []
for ids, pos, kind, lp in ctx:
    gp = pos[pos >= 160]
    if kind == "loop": gp = np.intersect1d(gp, np.flatnonzero(lp))
    if len(gp) == 0: continue
    e = int(rng.choice(gp)); prompts.append((kind, tok.decode(ids[e - 160:e])))
order = rng.permutation(len(prompts))[:N]; prompts = [prompts[i] for i in order]
print({k: sum(p[0] == k for p in prompts) for k in ["loop", "math", "nonmath", "wiki"]}, flush=True)
for p in prompts[:3]: print(repr(p[1][:300]))
jm = jlens.from_hf(m, tok)
lens = jlens.fit(jm, [p[1] for p in prompts], dim_batch=int(os.environ.get("JL_DIM_BATCH", 32)), max_seq_len=128, checkpoint_path=OUT + ".ckpt", checkpoint_every=25)
lens.save(OUT); print(lens)
