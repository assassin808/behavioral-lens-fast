#!/bin/bash
# Portable runner for test 5 (path patching). cwd = a checkout of branch yang/t5. No site-specific paths needed:
#   BLF_MODEL_BASE / BLF_MODEL_P50C  HF-format model dirs (p50c already rebuilt; exactly one *.safetensors each)
#   BLF_DATA                          cache dir for large per-sample outputs (.pt, not committed)
#   BLF_PY                            python (transformers>=5.11, scipy)
# Stages: smoke (2 texts x 1024) -> pilot (10 x 4096) -> automatic gate -> full (T5_N texts x 4096, default 20).
# Outputs (named by cluster): notes/test5_paths_<stage>_<cluster>.json + notes/test5_<stage>_<cluster>.log, committed locally.
#   STAGES="smoke pilot full" (default) — set e.g. STAGES=full to skip the earlier ones.
set -euo pipefail
export BLF_MODEL_BASE=${BLF_MODEL_BASE:-/scratch/yang0531/blf_models/base}
export BLF_MODEL_P50C=${BLF_MODEL_P50C:-/scratch/yang0531/blf_models/p50c}
export BLF_DATA=${BLF_DATA:-/home/yang0531/.claude/jobs/dcb4a01d/tmp/t5data}
PY=${BLF_PY:-/scratch/yang0531/resft/env/bin/python}
CL=${CC_CLUSTER:-$(hostname -s | sed 's/[0-9]*$//')}
N=${T5_N:-20}; STAGES=${STAGES:-"smoke pilot full"}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$BLF_DATA" notes
echo "cluster=$CL base=$BLF_MODEL_BASE p50c=$BLF_MODEL_P50C py=$PY"
run() {  # stage n maxt
    local tag="$1_${CL}"
    "$PY" scripts/test5_paths.py "$2" "$3" "$tag" 2>&1 | tee "notes/test5_$tag.log"
}
for s in $STAGES; do
    case $s in
        smoke) run smoke 2 1024 ;;
        pilot) run pilot 10 4096
               # gate: patch identity reproduces the unpatched metric, and the edit raises loop NLL by 0.03..0.3 (Q4: +0.093)
               "$PY" - "notes/test5_paths_pilot_${CL}.json" <<'EOF'
import json, sys
S = json.load(open(sys.argv[1]))
ok = S["id_err"] < 1e-3 and 0.03 < S["total"] < 0.3
print(f"GATE id_err={S['id_err']:.1e} total={S['total']:+.4f} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
EOF
               ;;
        full) run full "$N" 4096 ;;
    esac
    git add notes/test5_*"${CL}"* 2>/dev/null || true
    git -c user.name="${GIT_NAME:-blf-runner}" -c user.email="${GIT_EMAIL:-blf-runner@users.noreply.github.com}" \
        commit -q -m "yang/t5: test 5 $s results on $CL

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
done
echo "all stages done"
