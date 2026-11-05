#!/bin/bash
# Killarney wrapper for scripts/portable_t38.sh (fallback run; results tagged 'killarney').
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=kn101,kn104
cd /scratch/yang0531/blf
export BLF_MODEL_BASE=/scratch/yang0531/blf_models/base BLF_MODEL_P50C=/scratch/yang0531/blf_models/p50c BLF_DATA=/scratch/yang0531/blf_data
export BLF_PY=/scratch/yang0531/resft/env/bin/python BLF_VLLM_PY=/scratch/yang0531/resft/env/bin/python BLF_RUNTAG=killarney
export PYTHONPATH=/scratch/yang0531/blf_models/pylib HF_HOME=/scratch/yang0531/noise_pilot/hf_cache OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
bash scripts/portable_t38.sh   # T38_STAGE=pilot | full (separate 3h jobs on the interactive partition)
