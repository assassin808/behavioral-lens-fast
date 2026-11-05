#!/bin/bash
# Tests 3 + 8 GPU pipeline (branch yang/t3-t8), portable: no cluster-specific paths.
# Env: BLF_MODEL_BASE, BLF_MODEL_P50C (HF model dirs; p50c rebuilt with apply_edit.py), BLF_DATA (scratch for large outputs),
#      BLF_PY (python with torch + transformers>=5.11 [+ flash-linear-attention], sklearn, matplotlib),
#      optional BLF_VLLM_PY (python with vllm, for the thinking-off generations; else transformers fallback),
#      BLF_RUNTAG (suffix of committed result files; default = cluster name), T38_MAXT (default 32768 = full length), T38_NPC (default 100).
# cwd = a checkout of yang/t3-t8. Writes notes/test3_*, notes/test8_*, notes/fig/test8_* and commits them locally (no push).
set -euo pipefail
PY=${BLF_PY:-python}; VPY=${BLF_VLLM_PY:-$PY}
export T38_DATA=${BLF_DATA:?set BLF_DATA}/t38 PYTHONPATH=$PWD/scripts${PYTHONPATH:+:$PYTHONPATH} VLLM_USE_FLASHINFER_SAMPLER=0
export BLF_RUNTAG=${BLF_RUNTAG:-${CC_CLUSTER:-$(hostname -s | sed 's/[0-9]*$//')}}
MAXT=${T38_MAXT:-32768}; NPC=${T38_NPC:-100}
mkdir -p "$T38_DATA" notes/fig
LOG=notes/test38_run_${BLF_RUNTAG}.log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) runtag=$BLF_RUNTAG MAXT=$MAXT NPC=$NPC base=$BLF_MODEL_BASE p50c=$BLF_MODEL_P50C"
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
$PY -c "import fla; print('fla', fla.__version__)" || echo "no flash-linear-attention: torch fallback (slower)"
T0=$(date +%s)

# 1. base thinking-OFF generations (400 problems)
if [ ! -s "$T38_DATA/gen_base_nothink.jsonl" ]; then
  if $VPY -c "import vllm" 2>/dev/null; then GEN_BACKEND=vllm $VPY scripts/test8_gen_nothink.py
  else echo "vllm not importable -> transformers fallback (8k cap, no presence penalty)"; GEN_BACKEND=hf $PY scripts/test8_gen_nothink.py; fi
fi
$PY - <<'PYEOF'
import json, os
G = [json.loads(l) for l in open(os.environ["T38_DATA"] + "/gen_base_nothink.jsonl")]
print("no-think gens:", len(G), "cut", sum(g["cut"] for g in G), "loop_frac>0.05", sum(g["loop_frac"] > .05 for g in G))
for g in G[:2] + G[200:202]: print("--- sample", g["id"], g["n_tokens"], "tokens:", g["text"][:300].replace("\n", " | "), "...", g["text"][-200:].replace("\n", " | "))
PYEOF

# 2. pilot (8 contexts, 16k) + sign gate   (T38_STAGE=pilot: stop after it; T38_STAGE=full: skip it)
STAGE=${T38_STAGE:-all}
if [ "$STAGE" != full ]; then
PILOT=1 $PY scripts/test3_collect.py 16384 8
PILOT=1 $PY scripts/test8_tf.py 16384 8
$PY - <<'PYEOF'
import os, torch, numpy as np, sys
D = torch.load(os.environ["T38_DATA"] + "/t3_collect_16384_pilot.pt", weights_only=False)
lo = [(s["resid_u_16"][torch.from_numpy(s["m"])].float().mean().item(), s["resid_u_16"][torch.from_numpy(~s["m"] & (np.arange(len(s["m"])) >= m["P"]))].float().mean().item()) for s, m in zip(D["series"], D["meta"]) if m["cls"] == "loop"]
print("pilot resid.u16 (loop, non-loop) in loop texts:", [tuple(round(x, 3) for x in t) for t in lo])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get("BLF_MODEL_BASE", "/scratch/yang0531/blf_models/base"))
ws = sorted([w for w in D["windows"] if w["layer"] == 15], key=lambda w: -w["val"])[:8]
for w in ws: print(f"L15.{w['unit']} c={w['val']:.3f} loop={w['in_loop']}:", repr(tok.decode(w["ids"])))
if not (len(lo) and np.mean([a - b for a, b in lo]) > 0): print("PILOT GATE FAILED: u not higher at loop positions"); sys.exit(1)
print("pilot gate passed")
PYEOF
echo "pilot done $(( $(date +%s) - T0 ))s"
fi
[ "$STAGE" = pilot ] && exit 0

# 3. full
$PY scripts/test3_collect.py $MAXT $NPC
$PY scripts/test3_analyze.py $MAXT
$PY scripts/test3_explain_prep.py $MAXT
$PY scripts/test8_tf.py $MAXT $NPC
$PY scripts/test8_analyze.py $MAXT
echo "full done $(( $(date +%s) - T0 ))s"
git add notes/test3a_ranking_*_${BLF_RUNTAG}.json notes/test3_exemplar*_${BLF_RUNTAG}.* notes/test8_results_*_${BLF_RUNTAG}.json \
        notes/fig/test8_*_${BLF_RUNTAG}.png "$LOG" 2>/dev/null || true
cp "$T38_DATA"/gen_base_nothink_summary.json notes/test8_gen_nothink_summary_${BLF_RUNTAG}.json && git add notes/test8_gen_nothink_summary_${BLF_RUNTAG}.json
git -c user.name="${GIT_NAME:-blf-runner}" -c user.email="${GIT_EMAIL:-blf-runner@localhost}" commit -qm "t3/t8 GPU results ($BLF_RUNTAG, MAXT=$MAXT, NPC=$NPC)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1
echo "Next (login node): bash scripts/portable_t38_api.sh   (needs VECTOR_BASE_URL / VECTOR_API_KEY, BLF_DATA, BLF_PY)"
