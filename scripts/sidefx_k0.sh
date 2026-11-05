#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sidefx_k0
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/k0_%j.out
cd /scratch/yang0531/blf_sidefx
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
O=/scratch/yang0531/blf_data/sidefx
$PY scripts/sidefx_check.py /scratch/yang0531/blf_models/base notes/sidefx/specs/p50chook.json $O/check_hook_v2.npy || exit 1
$PY -c "
import numpy as np
O='$O'; p=np.load(f'{O}/check_p50c.npy'); b=np.load(f'{O}/check_base.npy'); h=np.load(f'{O}/check_hook_v2.npy')
db, dh = np.abs(b-p).mean(), np.abs(h-p).mean(); print(f'K0v2 mean|base-p50c|={db:.4f} mean|hook-p50c|={dh:.4f} ratio={dh/db:.3f} median={np.median(np.abs(h-p)):.5f} max={np.abs(h-p).max():.3f}', flush=True)
"
