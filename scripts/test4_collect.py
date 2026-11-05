"""Test 4: collect BASE-model residual activations hidden_states[16], [18], [20] for SAE training/analysis.

Corpus: all Astra generations with known prompts (unedited + p50c; gsm8k/math500/ifeval/mmlupro), prompt+generation <= MAXT
tokens, run through the base model; plus WikiText-103 1024-token chunks (normal non-chat text).
Output dir OUT: acts_L{16,18,20}.f16 (N x 1024 memmap), tok.i32, meta.npz (item index, pos, loop flag, split, source table).

    python scripts/test4_collect.py OUT [MAXT=16384] [WIKI_TOKENS=6000000] [LIMIT_ITEMS]
"""
import os, sys, time, json, numpy as np, torch
from test24_common import *

OUT = sys.argv[1]; MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
WIKI = int(sys.argv[3]) if len(sys.argv) > 3 else 6_000_000
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
LAYERS = [16, 18, 20]
os.makedirs(OUT, exist_ok=True)
assert not os.path.exists(f"{OUT}/meta.npz"), "refusing to overwrite"
m, tok = load_model("base")
items = chat_items(tok, maxt=MAXT)
wk = wiki_docs(WIKI, tok, seq=1024)
for i, w in enumerate(wk):
    items.append(dict(id=f"wiki_{i}", ds="wiki", gen_by="text", ids=w, P=0, loop=np.zeros(len(w), bool), finished=True,
                      split="test" if i % 100 < 15 else "train"))
if LIMIT: items = items[:LIMIT // 2] + items[-LIMIT // 2:]
N = sum(len(x["ids"]) for x in items)
print("items", len(items), "tokens", N, flush=True)
mm = {L: np.lib.format.open_memmap(f"{OUT}/acts_L{L}.npy", mode="w+", dtype=np.float16, shape=(N, 1024)) for L in LAYERS}
tokarr = np.zeros(N, np.int32); item_ix = np.zeros(N, np.int32); pos = np.zeros(N, np.int32); loop = np.zeros(N, bool)
off, t0 = 0, time.time()
with torch.no_grad():
    for k, x in enumerate(items):
        ids = torch.tensor(x["ids"][None].astype(np.int64)).cuda()
        o = m.model(ids, output_hidden_states=True)
        n = ids.shape[1]
        for L in LAYERS: mm[L][off:off + n] = o.hidden_states[L][0].half().cpu().numpy()
        tokarr[off:off + n] = x["ids"]; item_ix[off:off + n] = k; pos[off:off + n] = np.arange(n); loop[off:off + n] = x["loop"]
        off += n
        if k % 50 == 0: print(k, off, f"{off / (time.time() - t0):.0f} tok/s", flush=True)
for L in LAYERS: mm[L].flush()
tab = [{kk: x[kk] for kk in ["id", "ds", "gen_by", "P", "finished", "split"]} for x in items]
np.save(f"{OUT}/tok.npy", tokarr)
np.savez(f"{OUT}/meta.npz", item=item_ix, pos=pos, loop=loop)
json.dump(tab, open(f"{OUT}/items.json", "w"))
print("done", off, "sec", time.time() - t0)
