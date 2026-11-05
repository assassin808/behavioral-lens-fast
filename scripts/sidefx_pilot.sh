#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sidefx_pilot
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/pilot_%j.out
cd /scratch/yang0531/blf_sidefx
export PY=/scratch/yang0531/resft/env/bin/python
export PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
B=/scratch/yang0531/blf_models/base; P=/scratch/yang0531/blf_models/p50c; S=notes/sidefx/specs; O=/scratch/yang0531/blf_data/sidefx
nvidia-smi --query-gpu=name --format=csv,noheader; date
# K0 checked separately (scripts/sidefx_k0.sh)
for arm in p50c p50chook clampp5 loopsel clampmean a3 base; do
  case $arm in base) M=$B; SP="";; p50c) M=$P; SP="";; *) M=$B; SP="--spec $S/$arm.json";; esac
  $PY scripts/sidefx_eval.py $M pilot_$arm $SP --n 10 --samples 1 --outdir $O/pilot 2>&1 | grep -E "HOOKS|SUMMARY|Error|error" ; date
done
