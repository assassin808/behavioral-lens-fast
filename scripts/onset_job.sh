#!/bin/bash
# Loop-onset job wrapper (CPU or GPU). Submit from /scratch/yang0531:
#   sbatch --account=aip-xli135 ... -o /scratch/yang0531/blf_data/onset/logs/%x_%j.log -J NAME blf_onset/scripts/onset_job.sh "cmd1" ...
set -uo pipefail
cd /scratch/yang0531/blf_onset
export BLF_DATA=/scratch/yang0531/blf_data VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx NPROC=${NPROC:-16}
PY=/scratch/yang0531/resft/env/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
T0=$(date +%s)
for c in "$@"; do echo "=== $c"; $PY $c 2>&1 | grep -v -i "warn" ; echo "=== exit ${PIPESTATUS[0]} after $(( $(date +%s) - T0 ))s"; done
echo "JOB_SECONDS $(( $(date +%s) - T0 ))"
