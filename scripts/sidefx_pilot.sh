#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=1:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sidefx_pilot
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/pilot_%j.out
# Pilot gate K1: 10 GSM8K + 10 MATH-500 x 1 sample per arm; 7 arms run as concurrent processes on one GPU (eager, launch-bound).
cd /scratch/yang0531/blf_sidefx
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
B=/scratch/yang0531/blf_models/base; P=/scratch/yang0531/blf_models/p50c; S=notes/sidefx/specs; O=/scratch/yang0531/blf_data/sidefx
mkdir -p $O/pilot/logs; nvidia-smi --query-gpu=name --format=csv,noheader; date
for arm in p50c p50chook clampp5 loopsel clampmean a3 base; do
  case $arm in base) M=$B; SP="";; p50c) M=$P; SP="";; *) M=$B; SP="--spec $S/$arm.json";; esac
  $PY scripts/sidefx_eval.py $M pilot_$arm $SP --n 10 --samples 1 --mem 0.12 --max-seqs 32 --outdir $O/pilot > $O/pilot/logs/$arm.log 2>&1 &
  sleep 75
done
wait; date
grep -h -E "^HOOKS|^SUMMARY|Traceback|Error" $O/pilot/logs/*.log | cut -c1-600
