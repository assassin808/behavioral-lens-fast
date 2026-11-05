# Side effects of Astra's p50c edit: overshoot (H1) or collateral push on non-loop re-reading (H2)? (branch `yang/sidefx`)

## Question
The edited model often finishes without `\boxed{}`, degenerates into whitespace until the cap, and its accuracy on completed
problems is lower. Two candidate causes:
- **H1 (overshoot).** At loop positions the edit pushes the residual far past the normal states along −u (cos −0.94, test 8). These
  off-manifold states cause the damage.
- **H2 (collateral push).** The 128 neurons also fire, more weakly, at ordinary non-loop positions (in L15 mostly when the model
  re-reads or self-checks; test 3). The edit therefore pushes −u during normal reasoning too, and that push causes the damage.

## Design (written before any generation; numbers from `scripts/sidefx_design.py` → `notes/sidefx_design.json`, CPU only)
Source: the test-3 teacher-forced base collection. It has 100 loop contexts (≤ 32k tokens) and 34 thinking-on finished
normal contexts. c_j = act_j·uᵀw_j (base). The p50c push at a position is 6·Σ_j c_j.

| layer | resid·u after layer: normal mean / p5 | loop mean | p50c push at loop pos. (mean) | p50c push at normal pos. (mean) | normal positions pushed below p5 by p50c |
|---|---|---|---|---|---|
| L15 | 0.74 / 0.09 | 2.26 | 2.51 → edited loop mean −0.25 | 0.49 | 28 % |
| L19 (ignores the upstream L15 change) | 1.62 / 0.35 | 4.41 | 4.47 → −0.06 | 0.66 | 17 % |

Arms. All arms use one harness, `scripts/sidefx_eval.py`. Every arm runs vLLM with torch.compile off, so PyTorch hooks execute.
The prompt, sampling (think35), grader and loop rule are Astra's, exactly as in `screen_eval.py`. Thinking is on and the cap is
16k tokens. Variants are hooks on the **base** weights. The hook adds −Σ_j α_j act_j (uᵀw_j)·u at the layer output, which is
algebraically the rank-1 edit.

| arm | what | loop-position push (L15 / L19) | normal-position push (L15 / L19) | targets |
|---|---|---|---|---|
| base | real base weights, no hook | 0 | 0 | reference |
| p50c | real p50c weights, no hook | 2.51 / 4.47 | 0.49 / 0.66 | reference |
| **clampp5** | α = 6, all 128 columns; the edit may not push resid·u below the **5th percentile of normal states** (F = 0.090 / 0.346). Rule: p_new = max(p + d, min(p, F)). **Not expressible as a weight edit** (nonlinear). | 2.15 / 3.97 (all of the below-normal part removed: 0 % of states below p5 vs 88 % / 76 %) | 0.43 / 0.60 (88 % / 91 % of p50c kept) | **H1 only** |
| **loopsel** | loop-selective subset: neurons with mean c(loop) − 5·mean c(normal) > 0 (31 / 35 neurons). α raised to 9.68 / 8.03 so that the mean loop-position push **equals p50c's**. It is a pure rank-1 weight edit, applied here as a hook for harness parity. | 2.51 / 4.47 (= p50c) | 0.09 / 0.20 (18 % / 30 % of p50c) | **H2 only** |
| clampmean | α = 6, floor at the normal-state **mean** (F = 0.736 / 1.619). This is the "push only down to normal" variant. Hook only. | 1.52 / 2.79 | 0.16 / 0.23 | H1 + H2 jointly |
| a3 | α = 3, all 128 columns (weight-expressible) | 1.25 / 2.24 | 0.24 / 0.33 | simple dose baseline |

Caveats on the design:
- The design numbers use per-position means from teacher forcing on base texts. In generation the arms visit different states.
- The L19 numbers ignore the upstream L15 change.
- H2 is operationalized as the edit's push at non-loop positions. Test 3 links this push to "re-reading / self-check" firing (half of
  L15's descriptions), but this experiment does not tag re-reading spans.

## Metrics (shared rule-based classes, `classify` in `sidefx_eval.py` = `failure_classes_from_logs.py`)
- **completion**: share of samples not cut. **loop_end**: Astra's loop rule. Together these are the manipulation check (are loops still broken?).
- **SE1 no-box rate** = finished without a parseable `\boxed{}` / finished.
- **SE2 whitespace rate** = cut with a whitespace-only last 2,000 characters / all samples.
- **SE3 accuracy on completed**, plus **paired accuracy**: accuracy on problems where both p50c and the arm finish at least one sample.
  This is needed because acc-on-completed is confounded by which problems get completed.
- Problem-level bootstrap 95 % CIs for the arm − p50c differences.

## Pre-registered predictions and kill rules (written 2026-11-05 before any run; not edited afterwards)
- **K0 (harness kill).** Hook α = 6 on base must reproduce real p50c. Teacher-forced token log-probs on 8 p50c texts × 3k tokens:
  mean |Δlogprob(hook − p50c)| ≤ 0.1 × mean |Δlogprob(base − p50c)|. Otherwise stop and fix the hook. No generation runs before this passes.
- **K1 (pilot gate).** 10 GSM8K + 10 MATH-500 problems × 1 sample per arm. Read the raw outputs. Scale up only if:
  - the hook arms produce coherent text;
  - p50c breaks loops (loop_end ≤ 0.2);
  - the classes look right by eye (a "no-box" sample really has no box; a "whitespace" sample really is whitespace).
- **P1 (manipulation).** loopsel and clampp5 keep the loops broken: loop_end ≤ p50c + 0.05 on each set.
  If an arm fails this, its side-effect numbers are reported but **cannot adjudicate** H1/H2. Fewer completions means a different selection.
- **P2 (H2).** loopsel cuts SE1 (no-box rate among finished, both sets pooled) to ≤ 0.5 × p50c's, and SE2 to ≤ 0.5 × p50c's.
- **P3 (H1).** clampp5 cuts SE2 (whitespace) to ≤ 0.5 × p50c's. SE1 changes by less than ±30 % relative.
  My prior is that whitespace degeneration is an off-manifold effect and that no-box is collateral.
- **P4 (H2, accuracy).** Paired accuracy of loopsel ≥ p50c + 0.03 (pooled).
- **P5 (dose).** a3 and clampmean break fewer loops: loop_end ≥ p50c + 0.10 on GSM8K. Their per-finished side-effect rates are ≤ p50c's.
- **Reading rule.**
  - H1 is supported if clampp5 passes P1 and P3.
  - H2 is supported if loopsel passes P1 and P2.
  - Both or neither are possible outcomes, and either is reported as such.

## Outcome
(to be filled after the runs; predictions above stay unedited)

### K0 (harness check), log of what happened
- **Run 1** (hook from the formula, fp32). mean |Δlogprob| hook − p50c = 0.0101 vs base − p50c = 0.0911, ratio **0.111**.
  This **fails** the pre-registered ≤ 0.1 by a hair. Median |Δ| 0.0005, max 0.21 nats. Summed log-prob over 24k tokens: hook
  −15,906.4, p50c −15,908.2, base −16,063.5.
- **Fix attempt** (the hook now uses the exact bf16-rounded edited columns, dW = bf16(w − 6uuᵀw) − w). Ratio **0.112**, unchanged.
  So the residual is not a weight mismatch. It comes from the computation path: the base layer output is rounded to bf16, then the
  delta is added, whereas p50c does a single bf16 matmul with the edited weights.
- **Deviation from the pre-registered plan.** I added a **p50chook** generation arm (α = 6, all columns, through the same hook path).
  It is the primary comparator for the hook arms (clampp5, loopsel, clampmean, a3). Those comparisons then share the numerical
  path exactly. Real-weight p50c is still reported. The predictions above are unchanged; "p50c" in P1–P5 is read as p50chook for
  the hook arms, and p50chook vs p50c is reported as a harness check.

### Throughput note (harness)
- **Eager mode.** Eager vLLM at batch 256 on an L40S gives 5.4k tok/s, close to the earlier RunPod screening's 5.0k with CUDA graphs.
  At batch 10–20 it is launch-bound: 20 % GPU utilization, about 25 steps/s.
- **CUDA graphs not used.** `full_decode_only` CUDA graphs run 8.4k tok/s with the hooks installed. I could not verify that the hooks
  are captured correctly: greedy texts diverge between eager and graph mode even for the same arm, as expected from numerics, so
  divergence is no test. I therefore kept **eager for every arm** (`scripts/sidefx_cgtest.py`, log `cg_7292687`).
- **Merged generate call.** Both sets now go through one `generate` call.

### Pilot gate K1 (10 GSM8K + 10 MATH-500 × 1 sample per arm; 7 arms as concurrent processes on one L40S)
| arm | compl | loop_end | no-box / finished | whitespace | cut after `</think>` | acc_all |
|---|---|---|---|---|---|---|
| p50c (weights) | 0.45 | 0.05 | 0/9 | 0 | 0.25 | 0.35 |
| p50chook | 0.60 | 0.05 | 0/12 | 0 | 0.10 | 0.30 |
| clampp5 | 0.60 | 0.15 | 1/12 | 1 | 0.10 | 0.35 |
| loopsel | 0.45 | **0.35** | 0/9 | **3** | 0.20 | 0.40 |
| clampmean | 0.50 | 0.25 | 0/10 | 0 | 0.05 | 0.40 |
| a3 | 0.45 | 0.30 | 0/9 | 1 | 0.15 | 0.35 |

The base pilot was cancelled before it finished (20 sequences decoding to 16k in eager mode, launch-bound). Base runs in screening on 100 + 100 × 2.

Read by eye:
- **Gate passed.** Text is coherent in all hook arms, p50c breaks loops (0.05), and the classes are right. Specifically:
  - Every "whitespace" sample really is runaway indentation: bullets indented further and further until about 1M trailing spaces at the cap.
  - The clampp5 no-box sample really ends "Final Answer: 180 miles." without a box.
  - loopsel's loop_end samples really are verbatim loops ("Wait, `(1.25, 4.5)` is the location of the dot." × N).
- **New (exploratory, not pre-registered).** At the 16k cap the dominant p50c failure is not "no box" but "cut, still reasoning".
  In 6 of 10 such p50c samples `</think>` had already been emitted, and the *answer* section kept ruminating until the cap. Examples:
  "Wait, I'll check the wording again. 'restart the download from the beginning.'…", and "Let's try to verify if there is a typo in
  the question." I added the rate "cut after `</think>`" to the analysis as an exploratory metric. It is presumably the 16k-cap
  version of 32k's "finished without \boxed{}".
- p50c's wrong finished answers re-interpret the problem: "muffins are not fresh duck eggs" (26 vs 18), and "the flock size acts as
  a distractor" (25 vs 20). The model over-questions the wording rather than failing to re-check it.
- **Early signal (n = 20, noisy).** loopsel has the same mean loop-position push as p50c but breaks far fewer loops (0.35), and
  it shows whitespace degeneration too.
