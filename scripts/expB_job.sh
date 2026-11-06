#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=expB_build
#SBATCH --output=/scratch/yang0531/blf_data/expB/build_full_%j.log
# Exp B build: CPU-only work (16 procs); GPU is requested only because all Killarney partitions are GPU partitions.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd /scratch/yang0531/blf_expB
/scratch/yang0531/sis_poc/env/bin/python scripts/expB_paraphrase.py build --tag full --procs 16
