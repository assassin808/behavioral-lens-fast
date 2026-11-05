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

### Screening logistics (deviation, recorded before any screening result was read)
- **Queue limits.** The interac QOS allows one job per user, and nodes have about one free L40S. Arms therefore run 2 at a time on
  one GPU (`scripts/sidefx_screen_multi.sh`).
- **Throughput.** With 16k contexts it is about 3k tok/s combined, lower than the 1k-context benchmark. Running every arm at
  200 + 200 × 2 would not fit the 3h job limit or the budget.
- **What ran.**
  - p50chook and loopsel: × 2 samples (job 7293441).
  - clampp5, p50c, clampmean and a3: × 1 sample (job 7294772).
  - base: 100 + 100 × 1 (job 7294773).
- **Analysis rule.** The main table uses **sample k = 0 only for every arm** (`SFX_K0=1`), on the same problems and the same
  seed-0 sampling stream. The × 2 numbers for p50chook and loopsel are reported separately as a check.

## Results (screening: first 200 GSM8K + first 200 MATH-500, thinking on, 16k cap, Astra prompt/sampling/grader, shared eager-vLLM harness; sample k = 0 of every arm, n = 400 per arm)

Script: `SFX_K0=1 python scripts/sidefx_analyze.py <dir> screen_ <out> ...`. Results: `notes/sidefx/screen_analysis_k0.json`.
Per-sample outputs are in `/scratch/yang0531/blf_data/sidefx/screen/*.jsonl`. They are too large for git.
Accuracy decomposition: **acc_all = completion × acc_on_completed**.

| arm | completion × acc_on_completed = acc_all | loop_end | whitespace (cut, all-whitespace tail) | no-box / finished | cut after `</think>` (answer phase never ends) | paired acc vs p50chook (n problems) | mean tokens |
|---|---|---|---|---|---|---|---|
| base (100 + 100, n = 200) | 0.095 × 0.947 = 0.100 | 0.890 | 0 | 0/19 | 0.000 | 1.000 / 1.000 (15) | 15,249 |
| p50c (real weights) | 0.578 × 0.835 = **0.482** | 0.052 | 16 (4.0 %) | 3/231 | 0.080 | 0.873 / 0.862 (189) | 11,969 |
| p50chook (α 6, hook) | 0.560 × 0.817 = **0.460** | 0.077 | 13 (3.3 %) | 2/224 | 0.080 | (reference) | 11,855 |
| **clampp5** (no state below normal p5) | 0.532 × 0.845 = **0.453** | **0.075** | **6 (1.5 %)** | 2/213 | 0.095 | 0.871 / 0.854 (178) | 12,040 |
| **loopsel** (loop-selective neurons, dose-matched) | 0.422 × 0.852 = 0.375 | **0.393** | 3 (0.7 %) | 5/169 | 0.268 | 0.880 / 0.901 (142) | 13,509 |
| clampmean (floor at normal mean) | 0.410 × 0.921 = 0.385 | 0.285 | 0 | 0/164 | 0.170 | 0.950 / 0.943 (141) | 13,667 |
| a3 (α 3) | 0.415 × 0.910 = 0.385 | 0.345 | 3 (0.7 %) | 1/166 | 0.070 | 0.924 / 0.924 (144) | 12,901 |

Per set (GSM8K / MATH-500):

| arm | completion | acc_all | loop_end | whitespace |
|---|---|---|---|---|
| base (100 + 100) | 0.11 / 0.08 | 0.11 / 0.09 | 0.88 / 0.90 | 0 / 0 |
| p50c | 0.815 / 0.340 | 0.650 / 0.315 | 0.030 / 0.075 | 7 / 9 |
| p50chook | 0.760 / 0.360 | 0.585 / 0.335 | 0.060 / 0.095 | 7 / 6 |
| clampp5 | 0.740 / 0.325 | 0.610 / 0.295 | 0.055 / 0.095 | 3 / 3 |
| loopsel | 0.570 / 0.275 | 0.475 / 0.275 | 0.290 / 0.495 | 2 / 1 |
| clampmean | 0.550 / 0.270 | 0.505 / 0.265 | 0.265 / 0.305 | 0 / 0 |
| a3 | 0.575 / 0.255 | 0.510 / 0.260 | 0.290 / 0.400 | 0 / 3 |

Checks:
- **Harness reproduces earlier numbers.** p50chook (× 2, n = 800): MATH-500 completion 0.358, acc 0.323, loop_end 0.078. Real-weight p50c
  in the earlier RunPod screening table: 0.34 / 0.31 / 0.09.
- **Hook p50c vs real-weight p50c** (k = 0): acc 0.460 vs 0.482, whitespace 13 vs 16, loop_end 0.077 vs 0.052. These are within
  sampling noise. They are not identical, because the bf16 path differs (K0).
- **Fisher exact tests on whitespace counts** (k = 0, 400 per arm):

  | arm | vs p50chook | vs p50c + p50chook pooled (29/800) |
  |---|---|---|
  | clampp5 | p = 0.16 | p = 0.045 |
  | loopsel | p = 0.02 | p = 0.002 |
  | a3 | p = 0.02 | p = 0.002 |
  | clampmean | p < 0.001 | p < 0.001 |

  With both samples: loopsel 8/800 vs p50chook 23/800, p = 0.01.

Raw excerpts (screening, k = 0):
- p50c, no box (gsm8k_test_115): `…If she buys for $10 and $16, she buys for $10 and $16.\n\nResult: 0` (finished, no `\boxed{}`).
- p50c, cut after `</think>` (gsm8k_test_28). The answer phase keeps re-checking the wording until the cap:
  `…One final check.\n"Before the end" -> "closer to the end".\nIf the end is 60.\nStop 2 is at 45.…`
- clampp5, whitespace: `<~200 spaces>*   But in the context of the whole sentence: "The third friend pressed ... 10 times more than the fourth friend."`
  Each new bullet is indented further, until about 1M spaces at the cap. It is the same pattern in every arm.
- clampmean, unbroken loop (gsm8k_test_11): `Wait.\n    $4368 + 3960$.\n    $4368 + 3960$.\n    $4368 + 3960$.\n    $8+0=8$.…` repeated.
- a3 (gsm8k_test_7): `Wait, why am I getting 180 minutes? … I will provide the answer 160 minutes.\n\nWait, let's check the math again.…`

## Outcome per prediction (predictions above unchanged)

| prediction | outcome |
|---|---|
| K0 hook = p50c (ratio ≤ 0.1) | **fail by a hair** (0.111; 0.112 after the fix). Residual is bf16 path noise. Handled by adding p50chook as the comparator (deviation, logged above) |
| K1 pilot gate | **pass** |
| P1 clampp5 loop_end ≤ ref + 0.05 per set | **pass** (0.055 / 0.095 vs 0.060 / 0.095) |
| P1 loopsel loop_end ≤ ref + 0.05 per set | **fail, decisively** (0.290 / 0.495 vs 0.060 / 0.095). By the pre-registered rule, loopsel **cannot adjudicate H2** |
| P2 (H2) loopsel SE1 and SE2 ≤ 0.5× | inadmissible (P1 failed). Raw: SE1 is *higher* (5/169 vs 2/224), SE2 lower (3 vs 13), cut-after-`</think>` far higher (0.27 vs 0.08) |
| P3 (H1) clampp5 SE2 ≤ 0.5× ref; SE1 within ±30 % | **SE2 pass as stated** (6 vs 13 = 0.46×), but weak evidence: p = 0.16 vs p50chook, p = 0.045 vs pooled p50c-type arms. SE1 "unchanged" (2/213 vs 2/224) is uninformative: no-box is at floor at the 16k cap (see caveats) |
| P4 loopsel paired acc ≥ ref + 0.03 | **fail** (0.880 vs 0.901, n = 142), and inadmissible anyway |
| P5 a3 and clampmean loop_end ≥ ref + 0.10 on GSM8K; per-finished side effects ≤ ref | **pass** (a3 0.290, clampmean 0.265 vs 0.060). Side effects lower: whitespace 0–3, no-box 0–1 |
| Reading rule | **H1 is weakly supported** for the whitespace side effect only. **H2 cannot be tested by neuron selection.** The loop-selective subset does not break loops, so the "collateral" neurons are part of the loop-breaking mechanism itself |

## What this says
1. **The side effects and the loop fix trade off along one axis.**
   - Every variant that reduces the push enough to remove whitespace degeneration also stops breaking loops: clampmean, a3 and
     loopsel all have loop_end 0.29–0.39 vs 0.08.
   - Completion falls from 0.56 to 0.41–0.42, and acc_all from 0.46 to 0.38.
   - The only variant that keeps loop-breaking is clampp5, which removes just the deepest overshoot (states pushed below the 5th
     percentile of normal states). It halves whitespace degeneration (3.3–4.0 % → 1.5 %) at no measurable cost: completion
     0.53, acc 0.45, paired acc equal.
   - So the overshoot is mostly *needed*. Pushing loop states down only to the normal mean does not break loops (clampmean).
     Only the extreme tail of the overshoot is dispensable, and that tail is linked to whitespace degeneration.
2. **H2 as posed is not separable by neurons.**
   - Matching the mean loop-position push with only the 31 + 35 loop-selective neurons (α ≈ 9.7 / 8.0) fails to break loops.
   - The neurons that also fire during ordinary re-reading and self-checking are part of what breaks loops. Hypothesis, untested:
     they fire at loop *onset*, before a verbatim 32-gram exists, and the edit must act there. Cheap test: onset-locked c_j per
     neuron class on the test-3 base loop texts.
3. **"Accuracy on completed drops" is selection, not damage.**
   - On problems that both arms finish, accuracy is the same in every arm: paired acc within ±0.02 of p50chook for all five arms.
   - The weaker arms have higher acc_on_completed (0.91–0.92 vs 0.82) only because they finish fewer, easier problems.
   - This is consistent with xuanqi's jointly-completed result (0.952 vs 0.952).
4. **At a 16k cap the no-box failure appears as "the answer phase never ends".**
   - Finished-without-box is about 1 % in every arm here, vs 21 % on MATH-500 at 32k (xuanqi).
   - What we see instead is 8 % of p50c samples that emit `</think>` and then keep re-checking the problem's wording until the cap.
   - This rate is the same for clampp5 (9.5 %) and a3 (7 %). It does not track overshoot. It is higher wherever loops survive
     (loopsel 27 %, clampmean 17 %, mostly loops inside the answer phase).

## Caveats
- **One sample per problem** in the main table (k = 0). Whitespace counts are small (6–16 events), so the clampp5 effect is
  p = 0.045 against pooled references and p = 0.16 against p50chook alone. Treat it as suggestive.
- **16k cap**, not Astra's 32k. The no-box side effect cannot be measured at this cap. Testing whether clampp5 reduces it needs 32k.
- **clampp5 and clampmean are not weight edits** (nonlinear floors, implemented as hooks). loopsel and a3 are rank-1 weight edits.
- **Floors and design numbers** come from base teacher-forced states (34 finished thinking-on texts). The L19 floor ignores the
  upstream L15 change.
- **Sample stream.** p50chook and loopsel ran with n = 2 per prompt, the other arms with n = 1. Sample k = 0 is not the same
  random stream across these settings, only the same distribution.
- **Hook vs weights.** The hook path differs from real weights by bf16 rounding (K0). For hook arms the comparator is p50chook.
- **Classes are rule-based.** "Wrong answers" were not judged by an LLM.

## GPU-hours (L40S, Killarney gpubase_interac)
| job | what | wall |
|---|---|---|
| 7292209, 7292351 | K0 checks (run 1 and v2) | 8 min |
| 7292422, 7292805 | pilots. The first was cancelled because 1 arm at a time was launch-bound; the second ran 7 concurrent arms | 36 min |
| 7292687 | CUDA-graph test | 6 min |
| 7293337 | single-arm job, cancelled when the QOS limit became clear | 3 min |
| 7293441 | screening wave 1: p50chook, loopsel × 2 | 98 min |
| 7294772 | screening: clampp5, p50c, clampmean, a3 × 1 | 100 min |
| 7294773 | screening: base 100 + 100 × 1 | 16 min |
| **total** | | **≈ 4.5 L40S GPU-h** (one GPU per job) |
