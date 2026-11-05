#!/bin/bash
# Usage: sbatch [opts] scripts/test38_job.sh <python script> [args...]   (runs from repo root with the shared env)
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=kn101,kn104
cd /scratch/yang0531/blf
set -a; . ~/.env; set +a; export HF_TOKEN=${hf_token:-$HF_TOKEN}
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf/scripts
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/yang0531/resft/env/bin/python "$@"
