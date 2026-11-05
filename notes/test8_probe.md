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
