#!/bin/bash
# Full-GPU queue: TF sweep (HF) -> variants -> screening evals (200+200 x 2 samples x 16k, Astra protocol) -> full MATH-500 32k x4 for base/p50c
cd /data/behavioral-lens-fast
R=/data2/ec2-user/models/qwen3.5-0.8b-astra
echo "[queue] start $(date)"
conda run -n vllm python scripts/q4b_tf_sweep.py > notes/q4b_tf_sweep.log 2>&1
echo "[queue] TF sweep done $(date)"
conda run -n vllm python scripts/build_variants.py sel_alpha1 sel_alpha4 sel_alpha8 top64_combined_alpha6 top64_dprime_alpha6 top64_uproj_alpha6 rand_dir00 rand_dir01 rand_dir02 rand_dir03 rand_dir04 2>&1 | grep -v -i warning
S="--n 200 --samples 2 --max-tokens 16384"
run() { conda run -n vllm python scripts/screen_eval.py "$@" 2>&1 | grep -E "^SUMMARY|Error|error|Traceback"; echo "[queue] $2 done $(date)"; }
run $R/p50c p50c $S
run $R/base base $S
run $R/base base_nothink $S --no-think
run $R/base base_rep1.1 $S --rep-penalty 1.1
run $R/base base_rep1.2 $S --rep-penalty 1.2
run $R/base base_budget4k $S --think-budget 4096
run $R/base base_budget8k $S --think-budget 8192
run $R/variants/sel_alpha1 sel_alpha1 $S
run $R/variants/sel_alpha2 sel_alpha2 $S
run $R/variants/sel_alpha4 sel_alpha4 $S
run $R/variants/sel_alpha8 sel_alpha8 $S
run $R/variants/top64_combined_alpha6 top64_combined_alpha6 $S
run $R/variants/top64_dprime_alpha6 top64_dprime_alpha6 $S
run $R/variants/top64_uproj_alpha6 top64_uproj_alpha6 $S
for k in 00 01 02 03 04; do run $R/variants/rand_dir$k rand_dir$k $S; done
run $R/variants/rand_cols rand_cols $S
run $R/variants/all_alpha1.5 all_alpha1.5 $S
run $R/p10c p10c $S
run $R/p25c p25c $S
# full MATH-500, Astra's exact setting (32k, 4 samples)
run $R/base base_full32k --sets math500 --n 500 --samples 4 --max-tokens 32768
run $R/p50c p50c_full32k --sets math500 --n 500 --samples 4 --max-tokens 32768
echo "[queue] all done $(date)"
