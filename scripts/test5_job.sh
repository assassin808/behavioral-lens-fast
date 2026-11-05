#!/bin/bash
# Usage (from /scratch/yang0531): sbatch [opts] blf_t5/scripts/test5_job.sh <python script> [args...]
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=kn101,kn104
cd /scratch/yang0531/blf_t5
set -a; . ~/.env; set +a; export HF_TOKEN=${hf_token:-$HF_TOKEN}
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/yang0531/blf_models/pylib T5_DATA=/home/yang0531/.claude/jobs/dcb4a01d/tmp/t5data
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/yang0531/resft/env/bin/python "$@"
