#!/bin/bash
# Generic GPU job for the Astra analyses on an Alliance cluster.
# Usage (from $BLF_HOME):  sbatch --account=def-XXXX --gres=gpu:h100:1 --time=3:00:00 repo/alliance/job.sh <branch> <runner.sh>
#   e.g. sbatch --account=def-XXXX --gres=gpu:h100:1 --time=3:00:00 repo/alliance/job.sh yang/t5 scripts/portable_t5.sh
# Narval: --gres=gpu:a100:1 . Each branch is checked out into its own worktree, results are committed there.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
set -euo pipefail
BLF_HOME=${BLF_HOME:-$HOME/scratch/blf_run}
BRANCH=$1; RUNNER=$2; shift 2
cd "$BLF_HOME"
module load StdEnv/2023 2>/dev/null || true; export PATH=$HOME/.local/bin:$PATH; unset PIP_CONFIG_FILE PYTHONPATH
source env/bin/activate
[ -f ~/.env ] && { set -a; . ~/.env; set +a; }
export HF_HOME=$BLF_HOME/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_USE_FLASHINFER_SAMPLER=0
# Paths every portable runner reads
export BLF_MODEL_BASE=$BLF_HOME/models/base BLF_MODEL_P50C=$BLF_HOME/models/p50c BLF_DATA=$BLF_HOME/data BLF_PY=$(which python)
mkdir -p "$BLF_DATA"
WT=$BLF_HOME/wt/${BRANCH//\//_}
[ -d "$WT" ] || git -C repo worktree add "$WT" "origin/$BRANCH" -B "$BRANCH"
cd "$WT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$RUNNER" "$@"
