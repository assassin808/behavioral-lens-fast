#!/bin/bash
# waits for the running GSM8K gen eval, then: TF sweep (HF), then screening evals (vLLM), sequentially.
cd /data/behavioral-lens-fast
R=/data2/ec2-user/models/qwen3.5-0.8b-astra
while pgrep -f run_gen_evals.sh > /dev/null; do sleep 60; done
echo "[queue] gen eval finished $(date)"
conda run -n vllm python scripts/q4b_tf_sweep.py > notes/q4b_tf_sweep.log 2>&1
echo "[queue] TF sweep done $(date)"
conda run -n vllm python scripts/build_variants.py sel_alpha1 sel_alpha4 sel_alpha8 top64_combined_alpha6 top64_dprime_alpha6 rand_dir00 rand_dir01 rand_dir02 2>&1 | grep -v -i warning
S="--n 100 --samples 1 --max-tokens 16384"
run() { conda run -n vllm python scripts/screen_eval.py "$@" 2>&1 | grep -E "^SUMMARY|Error|error|Traceback"; }
run $R/p50c p50c $S
run $R/base base $S
run $R/base base_nothink $S --no-think
run $R/base base_rep1.1 $S --rep-penalty 1.1
run $R/base base_rep1.2 $S --rep-penalty 1.2
run $R/base base_budget4k $S --think-budget 4096
run $R/base base_budget8k $S --think-budget 8192
run $R/variants/sel_alpha2 sel_alpha2 $S
run $R/variants/sel_alpha4 sel_alpha4 $S
run $R/variants/sel_alpha8 sel_alpha8 $S
run $R/variants/top64_combined_alpha6 top64_combined_alpha6 $S
run $R/variants/top64_dprime_alpha6 top64_dprime_alpha6 $S
run $R/variants/rand_dir00 rand_dir00 $S
run $R/variants/rand_dir01 rand_dir01 $S
run $R/variants/rand_cols rand_cols $S
run $R/variants/all_alpha1.5 all_alpha1.5 $S
echo "[queue] all done $(date)"
