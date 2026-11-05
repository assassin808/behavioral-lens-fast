#!/bin/bash
# Usage (from /scratch/yang0531): sbatch [--time ..] blf_t24/scripts/test24_job.sh <script.py> [args]
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=kn101,kn104
#SBATCH --output=/scratch/yang0531/blf_data/logs/%x_%j.out
cd /scratch/yang0531/blf_t24
set -a; . ~/.env; set +a; export HF_TOKEN=${hf_token:-$HF_TOKEN}
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/yang0531/blf_t24/scripts:${EXTRA_PYPATH}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/scratch/yang0531/bx/env_jlens/bin/python -u "$@"
