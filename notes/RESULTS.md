# What Astra's p50c edit does inside Qwen3.5-0.8B: results

> **One sentence:** The edit does not make the model reason better. It turns a group of "go back and re-read / re-check" neurons into an **anti-copy switch**: the model gets kicked out of its loops, but its internal state overshoots, causing side effects, and the result is no better than simply turning thinking off.

This file collects tests 2 / 3 / 4 / 5 / 8 and the "edited model with thinking off" screening. Where it conflicts with the earlier `q1`–`q6` notes, **this file is authoritative** (see §4).

---

## 1. The causal chain, from weights to behaviour

```
128 neurons in L15 / L19        ← fire when the model goes back to re-read the problem / re-check (test 3)
   │ after the edit they write a large vector along −u
   ▼
direction u: "doubt, go back and check" ↔ "wrap up, move on"   ← 5 lenses + SAE agree (tests 2, 4)
   │ changes the queries of downstream attention
   ▼
L19 / L23 attention: induction attention to the earlier copy of the token about to be repeated drops ~1/3   ← path patching (test 5)
   │ the repeated token becomes the most suppressed token in the vocabulary; MLPs L20–23 partly counteract
   ▼
behaviour: repeats still start, but are cut short (longest repeat per text: median 90 tokens vs 27,513 in base)
cost: the state overshoots along −u, out of normal reasoning (test 8) → no \boxed{}, whitespace degeneration, no help with thinking off
```

## 2. Question × object × tool × finding

| Question | Object | Tools | Finding | Test / file |
|---|---|---|---|---|
| Which neurons were edited? | the 128 edited neurons | rank all neurons by extra u-write inside loops; top-activating windows; auto-interp (Sonnet explainer + judge, Opus spot check on 24) | Astra picked mostly the right ones: overlap with the true top-64 is 44 (L15) / 36 (L19), chance 1. About half of L15 fire on re-reading the problem / restating / self-checking; only 21% / 5% of top windows are inside verbatim loops. | test 3: `test3_neurons.md` |
| What do they write? | direction u | logit lens, tuned lens, Neuronpedia J-lens / J++, R-lens (inject ±u, read the next token) | +u promotes Wait / maybe / Actually / 还是 / 不是; −u promotes `\n\n` / So / Final / `</think>`. The edit's final logit change correlates **0.98** with injecting −u at L15. | test 2: `test2_lenses.md` |
| What is it at the feature level? | residual-stream features | own SAEs after L15 and L19 (16k each) + public layer-17 SAE, Sonnet labels | Latents switched off: restating, self-doubt. Latents switched on: half "final answer", half off-distribution (FVU 0.13 → 0.5). | test 4: `test4_sae.md` |
| Which path carries the effect? | attention heads, MLPs | path patching (block / inject), AtP, induction-attention measurement; 20 texts (H100) + 30-text replication incl. 10 held out | 91% goes through L16–23 attention (96% held out); the edit sites' direct path is 3%; MLPs L20–23 counteract (blocking them makes the effect 55% larger). Mechanism: **copy suppression** on the query side, not a stop signal. | test 5: `test5_paths.md` |
| Where do the states go? | residual states | probe, difference-of-means direction, PCA, the edited model's own generations | Edited loop states overshoot past normal states along −u (cos −0.94) and form their own cluster. Thinking-off loops use the same u axis. | test 8: `test8_probe.md` |
| Does the edit help with thinking off? | generation | xuanqi's `screen_eval.py`, identical settings, same H100 | No: GSM8K 0.522 vs 0.560 base, MATH-500 0.487 vs 0.510 (paired −0.04 / −0.02, not significant). | `screen/nothink_p50c_runpod.md` |

## 3. Behaviour (200 problems per set × 2 samples, 16k cap)

| Config | GSM8K acc | MATH-500 acc | mean tokens (GSM8K / MATH) |
|---|---|---|---|
| base, thinking on | 0.147 | 0.072 | 14983 / 15630 |
| **p50c, thinking on** | 0.570 | 0.312 | 9964 / 13759 |
| **base, thinking off** | 0.555–0.560 | 0.510–0.512 | ~1500 / ~5500 |
| p50c, thinking off | 0.522 | 0.487 | 1468 / 5701 |
| base, thinking budget 4k / 8k (forced `</think>`) | 0.517 / 0.537 | 0.315 / 0.343 | — |

**Accuracy = completion rate × accuracy on completed.** Base with thinking on completes only 12%, but gets 94% of those right. The edit raises completion a lot, while accuracy on completed problems drops to about 0.75. New failure modes on MATH-500 (xuanqi): finishes without `\boxed{}` 21%, whitespace degeneration 5%, long but neither looping nor finishing 12%.

## 4. Corrections to the earlier notes (this section is authoritative)

- **`q3_who_reads_it.md` says "copy attention unchanged; not a copy-head story". That is wrong.**
  - Q3 measured the duplicate-token offset: attention from the current token to its own earlier occurrence. That really is unchanged; L23 h6 is such a head.
  - Copying uses the induction offset: attention to the earlier copy of the *next* token. There L23 attention drops 34%, and this is the edit's main path. See `test5_paths.md`.
- **`q1_what_changed.md` treats the low indices of the edited neurons as a feature of Astra's choice. That is wrong.** 92% of the *true* top-64 are also below index 1792, concentrated in 0–447. This is how the model is organised, not a sign that Astra searched only half. See `test3_neurons.md`.
- **"Only 10% along u at the end" does not mean u is unimportant.** That is a geometric decomposition; causally, the edit correlates 0.98 with injecting −u. See `test2_lenses.md`.

## 5. Lessons for the tools

- **Use difference-of-means, not probe weights, to find a direction.** A logistic-regression probe has cosine only 0.05–0.16 with u, yet resid·u alone separates loop vs normal at AUC 0.98. The probe weight is whitened and understates the relation.
- **AtP is only good for screening.** Its rank correlation with exact patching is 0.62–0.67, and it is off by 3–4× on the L15 MLP. Confirm with exact patching.
- **Measure the right attention offset.** The wrong one gives the opposite conclusion (§4).

## 6. Untested hypotheses

- Do the "re-reading" neurons cause the side effects (no `\boxed{}`, lost re-checks)? Running now on branch `yang/sidefx`.
- Why do the late MLPs counteract the edit?

## 7. Compute and reproduction

- **GPU:** about 3.2 L40S-h (Killarney) + 0.75 H100-h (RunPod).
- **Pre-registered predictions:** each test file lists them with pass / fail. Failures are reported as they are; predictions were not edited afterwards.
- **Reproduce on other clusters:** `alliance/setup.sh` + `alliance/job.sh <branch> scripts/portable_*.sh`.
- **Large outputs:** per-sample outputs are not in git; they are on Killarney in `/scratch/yang0531/blf_data/`.
- **Figures** (in `notes/fig/`):
  - `test2_*`: u variance share and axis
  - `test4_*`: SAE features
  - `test5_mediators_full_runpod.png`: path patching
  - `test8_state_pca_killarney.png`: state overshoot
  - `test8_onset_locked_killarney.png`: around loop onset
