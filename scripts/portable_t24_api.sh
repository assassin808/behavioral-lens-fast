#!/bin/bash
# Tests 4 auto-interp (LOGIN node: needs internet). cwd = checkout of yang/t2-t4 after portable_t24.sh finished (notes/test4_analyze_$TAG.json).
#   set -a; . ~/.env; set +a      # VECTOR_BASE_URL, VECTOR_API_KEY
#   BLF_TAG=<same tag as the GPU run> bash scripts/portable_t24_api.sh
# Sonnet labels + detection-scores every selected feature; Opus re-labels + re-scores a fixed random 20 (spot check).
set -euo pipefail
: "${VECTOR_BASE_URL:?}" "${VECTOR_API_KEY:?}" "${BLF_TAG:?}"
PY=${BLF_PY:-python}; T=$BLF_TAG
A=notes/test4_analyze_$T.json
[ -s notes/test4_labels_sonnet_$T.json ] || $PY scripts/test4_label.py $A notes/test4_labels_sonnet_$T.json --model claude-sonnet-5-5 > notes/test4_label_sonnet_$T.log 2>&1
[ -s notes/test4_labels_opus_$T.json ] || $PY scripts/test4_label.py $A notes/test4_labels_opus_$T.json --model claude-opus-5-5 --subset 20 > notes/test4_label_opus_$T.log 2>&1
$PY scripts/test4_summarize.py $A notes/test4_labels_sonnet_$T.json notes/test4_labels_opus_$T.json notes/fig/test4_$T > notes/test4_summary_$T.md
git add notes/ && git commit -q -m "t4 auto-interp labels + summary ($T)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1
