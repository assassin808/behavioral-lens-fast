#!/bin/bash
# Tests 2 (lenses on u) + 4 (SAE) GPU pipeline, portable (no cluster paths). cwd = checkout of branch yang/t2-t4.
# Needs: BLF_MODEL_BASE, BLF_MODEL_P50C (HF dirs; p50c rebuilt with apply_edit.py), BLF_DATA (big outputs), BLF_PY (python).
# One-time login-node prep first: scripts/portable_t24_prep.sh (jlens package, public lenses / SAE, WikiText shard).
# Optional: BLF_TAG (output suffix, default = cluster name), STAGES (default "pilot fit read collect sae analyze").
# Stages whose output already exists are skipped (resumable). Disk: ~130 GB in $BLF_DATA (fp16 activations, 3 layers x 20.4M tokens).
# Memory: SAE training loads one layer's activations (42 GB) into RAM -> submit with --mem=160G.
set -euo pipefail
: "${BLF_MODEL_BASE:?}" "${BLF_MODEL_P50C:?}" "${BLF_DATA:?}"
PY=${BLF_PY:-python}
export BLF_TAG=${BLF_TAG:-${CC_CLUSTER:-$(hostname -s | sed 's/[0-9-].*//')}}
export PYTHONPATH=$PWD/scripts:${PYTHONPATH:-} BLF_DATA BLF_MODEL_BASE BLF_MODEL_P50C
STAGES=${STAGES:-"pilot fit read collect sae analyze"}
T=$BLF_TAG; D=$BLF_DATA; mkdir -p notes/fig "$D/logs"
$PY -c "import jlens, pyarrow" 2>/dev/null || { echo "jlens/pyarrow missing: trying pip (needs internet; else run scripts/portable_t24_prep.sh on the login node)"; $PY -m pip install -q "git+https://github.com/anthropics/jacobian-lens.git@581d398" pyarrow; }
for f in "$D/ext/jlens/Qwen3.5-0.8B_jacobian_lens.pt" "$D/ext/jpp/Qwen3.5-0.8B_jpp_lens.pt" "$D/ext/sae_l17/sae_weights.safetensors" "$D/wikitext/train0.parquet"; do
  [ -s "$f" ] || { echo "missing $f: run scripts/portable_t24_prep.sh on the login node"; exit 1; }; done
has() { [[ " $STAGES " == *" $1 "* ]]; }
ts() { echo "[$(date +%H:%M:%S)] $*"; }
if has pilot && [ ! -s notes/test24_pilot_$T.log ]; then ts pilot; $PY -u scripts/test24_pilot.py 2>&1 | grep -v -i "warn" > notes/test24_pilot_$T.log; fi
if has fit; then
  [ -s $D/jlens_ours_$T.pt ] || { ts jlens-fit; $PY -u scripts/test2_fit_jlens.py $D/jlens_ours_$T.pt 400 > $D/logs/fit_jlens_$T.log 2>&1; }
  [ -s $D/tuned_lens_$T.pt ] || { ts tuned-fit; $PY -u scripts/test2_fit_tuned.py $D/tuned_lens_$T.pt > notes/test2_fit_tuned_$T.log 2>&1; }
fi
if has read && [ ! -s notes/test2_read_$T.json ]; then
  ts test2-read; $PY -u scripts/test2_read.py $D/tuned_lens_$T.pt $D/jlens_ours_$T.pt notes/test2_read_$T.json 40 20 20 8192 > $D/logs/test2_read_$T.log 2>&1
  $PY scripts/test2_summarize.py notes/test2_read_$T.json $D/tuned_lens_$T.pt notes/fig/test2_$T > notes/test2_summary_$T.md
fi
if has collect && [ ! -s $D/acts_base_$T/meta.npz ]; then ts collect; rm -rf $D/acts_base_$T; $PY -u scripts/test4_collect.py $D/acts_base_$T 16384 6000000 > $D/logs/collect_$T.log 2>&1; fi
if has sae; then
  for L in 16 20; do [ -s $D/sae_L${L}_$T.pt ] || { ts sae-L$L; $PY -u scripts/test4_train_sae.py $D/acts_base_$T $L $D/sae_L${L}_$T.pt 16384 64 1 > notes/test4_sae_train_L${L}_$T.log 2>&1; }; done
fi
if has analyze && [ ! -s notes/test4_analyze_$T.json ]; then ts analyze; $PY -u scripts/test4_analyze.py $D/acts_base_$T $D/sae_L16_$T.pt $D/sae_L20_$T.pt notes/test4_analyze_$T.json 40 8192 > $D/logs/analyze_$T.log 2>&1; fi
ts done
git add notes/ 2>/dev/null && git commit -q -m "t2-t4 portable run ($T): $STAGES

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "committed locally: $(git log --oneline -1)" || echo "nothing to commit"
