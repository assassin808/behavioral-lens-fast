# Test 8 — loop probe vs u, and the probe on the edited model's own generations (branch `yang/t3-t8`)

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)

- **P8a** |cos(probe direction mapped back to raw residual space, u)| at hidden_states[16] / [20] is **moderate, 0.2–0.6**
  (the probe uses more than u); the difference-of-means loop direction has cos ≥ 0.4 with u. resid·u alone gives loop-vs-non AUC ≥ 0.75.
- **P8b-1** In p50c's own thinking-on generations, verbatim 32-gram loop runs are rarer and **shorter** (median run length
  < half of base's) — the model enters repetition and is kicked out.
- **P8b-2** Pre-edit (hidden_states[15], input of L15) the base-trained probe still fires on p50c's repeated stretches
  (P(loop) > 0.5 on ≥ 50 % of p50c loop positions), while after the edit (hidden_states[16]+) it does not (< 0.3) — the model
  still "recognizes" loops; the edit inverts the flag downstream.
- **P8b-3** Base thinking-OFF loops light up the same u: resid·u(loop) / resid·u(non-loop) ≥ 1.5 at hidden_states[16], as in thinking-on loops.
- **P8c** In a 2D projection (PCA of base states), edited loop states move away from the base loop cluster along ≈ −u but
  do not land in the normal cluster.
- **P8d (thinking on vs off)** Over the first 1024 generated tokens of the same problems (before loops start), mean resid·u is
  higher with thinking on than off, but the gap is **< 25 %** of the loop-vs-non-loop gap — u is a loop flag, not a "thinking mode" axis.
  Kill/alt: a gap ≥ 50 % of the loop gap would mean u also encodes thinking mode.

## Setup
- **Probe.** Logistic regression, standardized inputs, C = 0.1, class-balanced. Fit on 40,000 base states (14,000 loop) sampled
  from 200 math contexts (100 loop + 100 normal, ≤ 32k tokens; corpus as in `notes/test3_neurons.md`). 5-fold context-grouped CV.
  The final probe is mapped back to the raw residual: w_raw = w_std / sd.
- **Passes.** Teacher-forced, ≤ 32,768 tokens incl. prompt.
  - p50c on 100 of its own thinking-on GSM8K/MATH generations from Astra's logs (84 finished).
  - Base on the same 100 texts.
  - Base and p50c on 50 base loop texts.
  - Base on 76 base thinking-off generations: all 56 with loop_frac > 0.02, plus 20 clean.
  - Base on the first 2,048 generated tokens of thinking-on vs thinking-off generations of the same 400 problems.
- **Thinking-off generations.** `scripts/test8_gen_nothink.py`: vLLM, Astra sampling (T 0.6, top-p 0.95, top-k 20,
  presence 1.5), 16k cap. GSM8K cut 2 %, MATH cut 16.5 %; mean 1,583 / 5,813 tokens. This matches the screening run.
- Scripts: `scripts/test8_tf.py`, `scripts/test8_analyze.py`. Results: `notes/test8_results_32768_killarney.json`.
- Figures:
  - `notes/fig/test8_state_pca_killarney.png`
  - `notes/fig/test8_onset_locked_killarney.png`
  - `notes/fig/test8_timeseries_{p50c_own,base_loop,base_nothink}_killarney.png`
  - `notes/fig/test8_think_vs_nothink_u_killarney.png`
- Note: hidden_states[15] (the input of L15) is identical in base and p50c for the same text, because layers 0–14 are unedited.
  "Pre-edit" readings are therefore the same in both models by construction. The models differ from hidden_states[16] on.

## 8a. Probe direction vs u

| hidden_states[L] | CV bal. acc | cos(w_raw, u) | cos(w_std, u) | cos(mean_loop − mean_non, u) | AUC of resid·u alone | share of probe-logit variance from the u-component |
|---|---|---|---|---|---|---|
| 15 | 0.968 | 0.05 | 0.08 | 0.54 | 0.972 | 0.69 |
| 16 | 0.964 | 0.06 | 0.13 | 0.67 | 0.978 | 0.74 |
| 19 | 0.964 | 0.09 | 0.16 | 0.66 | 0.976 | 0.73 |
| 20 | 0.962 | 0.10 | 0.16 | **0.72** | **0.978** | 0.76 |
| 24 | 0.961 | 0.16 | 0.16 | 0.23 | 0.929 | 0.56 |

- The probe's weight vector is nearly orthogonal to u (cos ≤ 0.16). Yet 69–76 % of the variance of its logit comes from the states'
  u-component, and resid·u alone separates loop from non-loop as well as the probe does (AUC 0.98 vs bal. acc 0.96).
- So u is the dominant *signal* direction, i.e. the difference of means (cos 0.67–0.72 after the edited layers). The probe weight
  is a whitened version of it (w ∝ Σ⁻¹Δμ) that also cancels high-variance nuisance directions, so cos(w, u) understates the relation.
- Recommendation: use resid·u or the difference of means, not the probe weight, as the "loop direction".

## 8b. The probe on generations

| pass (model on text) | n texts / tokens | loop frac (32-gram) | 32-gram runs per 1k tokens | run length median / mean / p90 | longest run per text, median | texts with a run ≥ 1k tokens |
|---|---|---|---|---|---|---|
| base on base thinking-on loop texts | 50 / 1.63 M | 0.84 | 0.47 | 12 / 1,801 / 81 | 27,513 | 100 % |
| p50c on its own thinking-on gens | 100 / 1.77 M | 0.15 | **3.19** | 15 / 70 / 95 | **90** | 16 % |
| base on base thinking-OFF gens | 76 / 0.71 M | 0.12 | 2.01 | 12 / 94 / 56 | 69 | 16 % |

P(loop) and resid·u at loop vs non-loop positions:

| pass | P(loop) hs15 at loop pos | P(loop) hs16 | P(loop) hs20 | resid·u hs16, loop / non | resid·u hs20, loop / non |
|---|---|---|---|---|---|
| base on base loop texts | 0.993 | 0.992 | 0.990 | 2.26 / 0.97 (2.3×) | 4.41 / 1.94 |
| p50c on the same texts | 0.993 (same by construction) | **0.66** | **0.014** | −0.25 / 0.33 | −0.00 / 0.85 |
| p50c on its own gens | 0.89 | 0.46 | **0.027** | −0.14 / 0.35 | 0.10 / 0.84 |
| base on p50c's own gens | 0.89 | 0.86 | 0.86 | 1.89 / 1.05 | 3.62 / 2.11 |
| base on thinking-OFF gens | 0.90 | 0.88 | 0.89 | **1.95 / 0.81 (2.4×)** | 3.72 / 1.62 |

- **The edited model still enters loops, often, and gets kicked out.** It starts a verbatim 32-gram repeat 3.2 times per 1k tokens,
  7× as often as the base model. But a run usually ends within a few dozen tokens: mean run 70 vs 1,801, and the longest run per
  text has median 90 vs 27,513.
- Upstream, the edited model's own states still say "loop": P(loop) at hs[15] is 0.89 on its loop positions, and the base model
  reads the same texts as loop at every layer (0.86). After the edited layers the flag is gone: P(loop) 0.03 at hs[20], and resid·u
  is *lower* at loop positions than elsewhere.
- Onset-locked averages (`test8_onset_locked_killarney.png`; 2,523 p50c onsets, 411 base, 918 thinking-off):
  - In base thinking-on loops, P(loop) and resid·u at hs[20] keep climbing after onset. The loop locks in.
  - In p50c, P(loop) at hs[20] stays flat at about 0.02, and resid·u at hs[20] stays flat at about 0.8, dipping right at onset.
  - In the time series (`test8_timeseries_p50c_own_killarney.png`), blue (hs15) spikes are cut to near 0 in red (hs20) in every
    episode. The episodes become denser late in the generation.
- **Thinking-off loops use the same u.** resid·u is 2.4× higher at loop positions (hs16). The base-trained probe (thinking-on
  states) reads them at 0.88–0.90, and onset-locked P(loop) shows the same spike. Thinking-off loops are short, like p50c's
  (longest run per text median 69; 16 % have a run ≥ 1k). So the base model without thinking also escapes most repeats, without
  any edit.

## 8c. Geometry (hidden_states[20], 3,000 states per group; `notes/fig/test8_state_pca_killarney.png`)

- PC1 of base normal + base loop states is essentially u: cos(PC1, u19) = 0.74.
- Mean resid·u per group:

  | group | mean resid·u |
  |---|---|
  | base normal | 1.30 |
  | base loop | 4.41 |
  | **edited loop states** (p50c on base loop texts) | **0.005** |
  | p50c own loop positions | 0.20 |
  | p50c own non-loop positions | 0.83 |

- The displacement from base loop to edited loop has cos −0.94 with u. The edit overshoots: edited loop states sit *below* normal
  states on u.
- Distance to the normal centroid: edited loop 5.09, base loop 5.92, while normal states themselves sit 5.75 from it.
- In the (u, top PC ⟂ u) plane, edited loop states form their own cluster, left of the normal cloud and shifted up on the
  orthogonal PC. They do not overlap it. p50c's own non-loop states sit between the two.
- So the edit moves loop states along −u past the normal manifold; they are not returned to normal reasoning. This replicates Q5
  with full-length contexts.

## 8d. Thinking on vs off (base, same 400 problems, non-loop positions only, first N generated tokens)

| N | hs | think | no-think | diff ± s.e. | % problems think > no-think | diff / loop gap (base thinking-on) |
|---|---|---|---|---|---|---|
| 256 | 16 | 0.49 | 0.41 | 0.086 ± 0.006 | 76 % | 0.07 |
| 1024 | 16 | 0.68 | 0.48 | 0.193 ± 0.006 | 95 % | 0.15 |
| 2048 | 16 | 0.77 | 0.51 | 0.264 ± 0.007 | 96 % | **0.20** |
| 2048 | 20 | 1.62 | 1.19 | 0.433 ± 0.012 | 96 % | 0.18 |

- Thinking-on states sit consistently higher on u than thinking-off states of the same problems, and the gap grows with position.
  In both modes resid·u drifts upward over the first 2k tokens; the thinking-on curve is higher
  throughout, by a roughly constant offset after ~250 tokens (`test8_think_vs_nothink_u_killarney.png`). The figure averages all
  positions over texts ≥ 2,048 tokens long: 388 thinking-on, 118 thinking-off. The table uses non-loop positions of all 400 problems.
- By 2k tokens the gap is 20 % of the loop gap. Thinking mode lives partway up the loop axis even before any verbatim repeat.
- Hypothesis (untested): the upward drift reflects a growing amount of re-reading and self-checking. That matches
  test 3's L15 neurons, which fire on referring back. Cheap test: regress resid·u on a re-reading tag per span.

## Outcome vs predictions (predictions above unchanged)

| prediction | outcome |
|---|---|
| P8a \|cos(probe_raw, u)\| 0.2–0.6; diff-of-means cos ≥ 0.4; AUC(resid·u) ≥ 0.75 | probe cos **fail** (0.05–0.16, lower than predicted); diff-of-means **pass** (0.54–0.72 at hs15–20; 0.23 at hs24); AUC **pass** (0.93–0.98) |
| P8b-1 p50c runs rarer and median run length < half of base | **fail as stated**: p50c runs are *more* frequent (3.2 vs 0.47 per 1k) and the median is 15 vs 12. Rerun with mean length (70 vs 1,801) or longest run (90 vs 27,513), they are far shorter; the pre-registered median was the wrong statistic (base median is dominated by short re-occurrences before lock-in) |
| P8b-2 hs15 P(loop) > 0.5 on ≥ 50 % of p50c loop positions; < 0.3 after the edit | **pass** (0.90 of positions > 0.5 at hs15; P(loop) 0.027 at hs20). hs16 alone is 0.46, so the flag is fully cleared only by L19. Caveat: hs15 equals the base reading by construction |
| P8b-3 thinking-off loops: resid·u loop/non ≥ 1.5 at hs16 | **pass** (2.4×) |
| P8c edited loop states move along −u, not into the normal cluster | **pass** (cos −0.94; overshoot below normal on u; a separate cluster in the u × PC⟂u plane) |
| P8d think − no-think gap < 25 % of the loop gap | **pass** (7–20 %), but the gap grows with position and is consistent (96 % of problems at 2k) |

Caveats:
- One sample per problem.
- p50c's own generations come from Astra's logs (32k thinking-on, Astra sampling). The thinking-off generations are ours (16k cap).
- The "loop" label is a 32-gram detector. Paraphrased loops are not counted.
- PCA uses 3,000 subsampled states per group.
