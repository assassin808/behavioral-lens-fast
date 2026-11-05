#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpubase_interac
#SBATCH --time=0:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sidefx_cg
#SBATCH --output=/scratch/yang0531/blf_data/sidefx/logs/cg_%j.out
cd /scratch/yang0531/blf_sidefx
export PY=/scratch/yang0531/resft/env/bin/python PYTHONPATH=/scratch/yang0531/blf_models/pylib:/scratch/yang0531/blf_models/pylib_sidefx MATHVERIFY_PKGS=/scratch/yang0531/blf_models/pylib_sidefx
export HF_HOME=/scratch/yang0531/noise_pilot/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
O=/scratch/yang0531/blf_data/sidefx; B=/scratch/yang0531/blf_models/base; S=notes/sidefx/specs
$PY scripts/sidefx_cgtest.py $B $S/clampp5.json full_decode_only $O/cg_clampp5_full.json 2>&1 | grep -E "HOOKS|CGTEST|Error|Traceback" 
$PY scripts/sidefx_cgtest.py $B $S/clampp5.json none $O/cg_clampp5_none.json 2>&1 | grep -E "HOOKS|CGTEST|Error|Traceback"
$PY scripts/sidefx_cgtest.py $B none full_decode_only $O/cg_base_full.json 2>&1 | grep -E "HOOKS|CGTEST|Error|Traceback"
$PY -c "
import json; O='$O'
a=json.load(open(f'{O}/cg_clampp5_full.json'))['ids']; b=json.load(open(f'{O}/cg_clampp5_none.json'))['ids']; c=json.load(open(f'{O}/cg_base_full.json'))['ids']
def agree(x,y):
  r=[]
  for p,q in zip(x,y):
    k=next((i for i,(s,t) in enumerate(zip(p,q)) if s!=t), min(len(p),len(q))); r.append(k)
  return r
print('CMP first-divergence hook full vs hook eager', agree(a,b)); print('CMP hook full vs base full', agree(a,c))
"
