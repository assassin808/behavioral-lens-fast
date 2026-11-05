#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/screen_%x_%j.out
# usage: sbatch -J sfx_<arm> scripts/sidefx_screen.sh <arm> [n] [samples]
# Screening: first n GSM8K + first n MATH-500, x samples, 16k cap, thinking on, eager vLLM (shared harness).
cd /scratch/yang0531/blf_sidefx
arm=$1; N=${2:-200}; K=${3:-2}
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
B=/scratch/yang0531/blf_models/base; P=/scratch/yang0531/blf_models/p50c; S=notes/sidefx/specs; O=/scratch/yang0531/blf_data/sidefx
case $arm in base) M=$B; SP="";; p50c) M=$P; SP="";; *) M=$B; SP="--spec $S/$arm.json";; esac
nvidia-smi --query-gpu=name --format=csv,noheader; date
$PY scripts/sidefx_eval.py $M screen_${arm}_n${N}x${K} $SP --n $N --samples $K --mem 0.85 --max-seqs 256 --outdir $O/screen 2>&1 | grep -vE "it/s|s/it"
date
