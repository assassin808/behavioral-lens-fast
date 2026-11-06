#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=3:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sfx_confirm
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/confirm_multi_%j.out
# usage: sbatch scripts/sidefx_confirm_multi.sh <slots> <mem per slot> <outsub> "arm set offset N K maxtok seed" ...
# Same as sidefx_screen_multi.sh (shared harness scripts/sidefx_eval.py, eager vLLM, hook spec per arm on base weights), plus
# set / offset / seed / cap per item and --save-ids. <slots> items run concurrently on ONE GPU, pulling from a queue.
cd /scratch/yang0531/blf_sidefx_confirm
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
B=/scratch/yang0531/blf_models/base; P=/scratch/yang0531/blf_models/p50c; S=notes/sidefx/specs
SLOTS=$1; MEM=$2; O=/scratch/yang0531/blf_data/sidefx/$3; shift 3
mkdir -p $O/logs $O/q; nvidia-smi --query-gpu=name --format=csv,noheader; date
Q=$O/q/queue_$SLURM_JOB_ID; printf "%s\n" "$@" > $Q; cat $Q
worker() {
  g=$1
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && break
    set -- $line; arm=$1; set_=$2; off=$3; N=$4; K=$5; MT=$6; SEED=$7
    case $arm in base) M=$B; SP="";; p50c) M=$P; SP="";; *) M=$B; SP="--spec $S/$arm.json";; esac
    tag=confirm_${arm}_${set_}_o${off}_n${N}x${K}_$((MT/1024))k_s${SEED}
    echo "[slot$g] start $tag $(date +%T)"
    $PY scripts/sidefx_eval.py $M $tag $SP --sets $set_ --offset $off --n $N --samples $K --max-tokens $MT --seed $SEED --save-ids \
        --mem $MEM --max-seqs 256 --outdir $O > $O/logs/${tag}.log 2>&1
    echo "[slot$g] end $tag rc=$? $(date +%T) $(grep -oE '"sec": [0-9]+, "tok_per_s": [0-9]+' $O/logs/${tag}.log)"
  done
}
for g in $(seq 1 $SLOTS); do worker $g & sleep 90; done
wait; date
