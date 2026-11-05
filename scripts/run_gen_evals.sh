#!/bin/bash
# sequential vLLM generation evals; one process per variant
cd /data/behavioral-lens-fast
R=/data2/ec2-user/models/qwen3.5-0.8b-astra
conda run -n vllm python scripts/build_variants.py sel_alpha2 all_alpha1.5 rand_cols 2>&1 | grep -v -i warning
for v in p50c base; do conda run -n vllm python scripts/gen_eval_gsm8k.py $R/$v $v 50 16384 2>&1 | grep -E "^SUMMARY|Error|error" ; done
for v in sel_alpha2 all_alpha1.5 rand_cols; do conda run -n vllm python scripts/gen_eval_gsm8k.py $R/variants/$v $v 50 16384 2>&1 | grep -E "^SUMMARY|Error|error" ; done
