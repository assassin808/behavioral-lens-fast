#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=3:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sfx_screen
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/screen_multi_%j.out
# usage: sbatch scripts/sidefx_screen_multi.sh <slots> <mem per slot> "arm N [K]" ... (K samples, default 2)
# interac QOS allows 1 job/user and nodes have ~1 free GPU: <slots> arms run concurrently on ONE GPU, pulling from a queue.
cd /scratch/yang0531/blf_sidefx
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
B=/scratch/yang0531/blf_models/base; P=/scratch/yang0531/blf_models/p50c; S=notes/sidefx/specs; O=/scratch/yang0531/blf_data/sidefx
mkdir -p $O/screen/logs $O/screen/q; nvidia-smi --query-gpu=name --format=csv,noheader; date
SLOTS=$1; MEM=$2; shift 2
Q=$O/screen/q/queue_$SLURM_JOB_ID; printf "%s\n" "$@" > $Q; cat $Q
worker() {
  g=$1
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && break
    set -- $line; arm=$1; N=$2; K=${3:-2}
    case $arm in base) M=$B; SP="";; p50c) M=$P; SP="";; *) M=$B; SP="--spec $S/$arm.json";; esac
    echo "[gpu$g] start $arm $(date +%T)"
    $PY scripts/sidefx_eval.py $M screen_${arm}_n${N}x${K} $SP --n $N --samples $K --mem $MEM --max-seqs 256 --outdir $O/screen > $O/screen/logs/${arm}.log 2>&1
    echo "[gpu$g] end $arm rc=$? $(date +%T) $(grep -oE '"sec": [0-9]+, "tok_per_s": [0-9]+' $O/screen/logs/${arm}.log)"
  done
}
for g in $(seq 1 $SLOTS); do worker $g & sleep 90; done
wait; date
