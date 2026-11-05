# Branch `xuanqi` — execution plan (merges the README's six questions on `main` with the task split below)

Shared conventions (adopted): Astra's prompt / sampling / grader via `scripts/astra_eval` (wraps `astra_logs/exp8_math/code/mathqa.py`
+ math-verify, installed at /data2/ec2-user/mathverify_pkgs); every generation reports completion rate, accuracy on problems both
sides completed, loop-end rate (`astra_eval.three_numbers`). Shared files in `shared/`: `u_L15.npy`, `u_L19.npy` (+u = loop
direction, cos 0.85), `astra_cols_L{15,19}.npy`, re-rankings `top64_by_{dprime,uproj}_L*.npy`, `top64_combined_L*.npy`.

**Compute reality on this box**: the Pro 6000 holds a Qwen3-Coder-30B vLLM server (88 GB). The 0.8B runs next to it with
gpu_memory_utilization 0.07 (3.4 GB KV cache, ~8 concurrent 32k sequences). Throughput is being measured by the running GSM8K eval;
the "5–10 min per config" estimates assume a free GPU and will not hold here. Screening set is sized from the measured tok/s.

| plan item | README Q | status | where |
|---|---|---|---|
| 1. recover u, re-rank neurons, index skew | Q1 | **done**. ΔW rank-1, w' = w − 6u(uᵀw). Index skew confirmed: Astra's columns have median index 364 / 354 (uniform 1792), 92 % / 78 % below the midpoint, MW p < 1e-3 — Astra scanned low indices. Overlap with true top-64: by d' 25 / 10, by uᵀw 31 / 18, combined 39 / 20. | `scripts/test1_shared_u.py`, `shared/test1_summary.json` |
| Finding 1 repro: completion vs conditional accuracy | Q6 | **done from logs** (no generation needed): acc on jointly-completed problems identical; all rescues were cut-offs. | `notes/finding1_completion_vs_accuracy.md` |
| 7. decoding baselines (thinking off, repetition penalty, thinking budget, u-steering) | Q4 | scripts ready, queued on GPU after the current eval | `scripts/screen_eval.py`, `scripts/queue_after_gen_eval.sh` |
| 6. causal controls: ≥20 random dirs, true top-64 vs Astra's, α 1/2/4/6/8 | Q4 | teacher-forced version (cheap) queued; generation version on the screening set after it | `scripts/q4b_tf_sweep.py`, `scripts/build_variants.py` |
| 2. logit / J / R-lens of u | Q1 | logit lens done (weak direct path; Q3b shows only 10 % of the final Δ along u). J/R-lens: collaborator | `notes/q1_what_changed.md` |
| 3. activation analysis of the 64 neurons | Q2 | done (loop-selectivity d', signed write along u); auto-interp exemplars: API | `notes/q2_when_does_it_act.md` |
| 8. loop probe vs u | Q5 | probe done (acc 0.91; edited loop states read as not-loop 0.85 → 0.08); cos(probe, u) still to add | `notes/q5_state_geometry.md` |
| 5. attribution / path patching (L15h7, L19h5, induction heads) | Q3 | first pass done: copy attention unchanged, Δ broadband, mediated by L22–23. Proper AtP: collaborator | `notes/q3_who_reads_it.md` |
| 4. SAE / crosscoder | — | P2, not started | |
| failure classification with an LLM judge | Q6 | not started; API candidates on this box: Vector proxy (Claude), Gemini | |

Open questions for you: collaborator's GPU; which API for auto-interp (Vector/Claude or Gemini are set up here).
