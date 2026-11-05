#!/bin/bash
# Test 3c auto-interp on a LOGIN node (needs internet): Sonnet explainer + judge on all 160 neurons, Opus spot check on 24, comparison.
# Env: VECTOR_BASE_URL, VECTOR_API_KEY (e.g. `set -a; . ~/.env; set +a`), BLF_DATA, BLF_PY, BLF_RUNTAG (same as the GPU run), T38_MAXT.
set -euo pipefail
PY=${BLF_PY:-python}; MAXT=${T38_MAXT:-32768}
export T38_DATA=${BLF_DATA:?set BLF_DATA}/t38 PYTHONPATH=$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}
export BLF_RUNTAG=${BLF_RUNTAG:-${CC_CLUSTER:-$(hostname -s | sed 's/[0-9]*$//')}}
: "${VECTOR_BASE_URL:?}" "${VECTOR_API_KEY:?}"
LOG=notes/test38_api_${BLF_RUNTAG}.log; exec > >(tee -a "$LOG") 2>&1
EXPLAINER=api:claude-sonnet-5-5 $PY scripts/test3_explain.py $MAXT all
EXPLAINER=api:claude-opus-5-5 $PY scripts/test3_explain.py $MAXT 8
$PY scripts/test3_explain_compare.py notes/test3_autointerp_${MAXT}_claude-sonnet-5-5_${BLF_RUNTAG}.json \
    notes/test3_autointerp_${MAXT}_claude-opus-5-5_sub8_${BLF_RUNTAG}.json claude-sonnet-5-5
git add notes/test3_autointerp_*_${BLF_RUNTAG}.json "$LOG"
git -c user.name="${GIT_NAME:-blf-runner}" -c user.email="${GIT_EMAIL:-blf-runner@localhost}" commit -qm "t3c auto-interp (Sonnet main, Opus spot check; $BLF_RUNTAG)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1
