#!/bin/bash
# Killarney job wrapper for experiment A. Usage (submit from /scratch/yang0531):
#   sbatch --account=aip-xli135 -p gpubase_interac --gres=gpu:l40s:1 --time=H:MM:00 --mem=64G -c 8 \
#          -o /scratch/yang0531/blf_data/expA/logs/%x_%j.log -J NAME blf_expA/scripts/expA_job.sh "cmd1" "cmd2" ...
set -uo pipefail
cd /scratch/yang0531/blf_expA
export BLF_DATA=/scratch/yang0531/blf_data VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export EXPA_SET=${EXPA_SET:-}
PY=/scratch/yang0531/resft/env/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
for c in "$@"; do echo "=== $c"; $PY $c 2>&1 | grep -v -i "warn" ; echo "=== exit ${PIPESTATUS[0]} after $(( $(date +%s) - T0 ))s"; done
echo "JOB_SECONDS $(( $(date +%s) - T0 ))"
