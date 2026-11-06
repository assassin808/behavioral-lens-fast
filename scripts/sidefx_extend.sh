#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=1:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sfx_ext32k
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/confirm_ext_%j.out
# usage: sbatch scripts/sidefx_extend.sh <arm> [<arm2>]   -- stage 2: extend cut MATH-500 k=0 samples to 32,768 total; arms run concurrently.
cd /scratch/yang0531/blf_sidefx_confirm
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
B=/scratch/yang0531/blf_models/base; S=notes/sidefx/specs; O=/scratch/yang0531/blf_data/sidefx/confirm
MEM=$(python3 -c "print(round(0.84/$#,2))"); nvidia-smi --query-gpu=name --format=csv,noheader; date
for arm in "$@"; do
  cat $O/confirm_${arm}_math500_o0_n250x2_16k_s1.jsonl $O/confirm_${arm}_math500_o250_n250x2_16k_s1.jsonl > $O/stage1_${arm}_math500_n500x2_16k_s1.merged.jsonl
  $PY scripts/sidefx_extend.py $B $O/stage1_${arm}_math500_n500x2_16k_s1.merged.jsonl confirm32k_${arm}_math500_n500k0_s2 --spec $S/$arm.json \
      --k 0 --total 32768 --seed 2 --mem $MEM --outdir $O > $O/logs/ext32k_${arm}.log 2>&1 &
  sleep 90
done
wait; grep -h SUMMARY $O/logs/ext32k_*.log; date
