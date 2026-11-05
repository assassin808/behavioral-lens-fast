# Q4 — Is a simpler intervention enough? (teacher-forced proxy)

Script: `scripts/q4_tf_interventions.py`. 6 unedited GSM8K generations with real loops (loop_frac 4–22 %), first 4096 tokens.
Metric: NLL of the actual next token at loop positions (how sure the model is of the repeated continuation) and at productive
positions (collateral damage). Base: loop NLL 0.021 (the loop is an attractor: the model is ~98 % sure of the next token),
productive NLL 0.505.

| variant | loop NLL | Δ loop | prod NLL | Δ prod |
|---|---|---|---|---|
| base | 0.021 | — | 0.505 | — |
| **p50c** (= sel_alpha6, exact reproduction) | **0.114** | **+0.093** | 0.558 | +0.053 |
| same 64 cols, α=1 (remove the u component) | 0.025 | +0.005 | 0.510 | +0.005 |
| same 64 cols, α=2 (flip it) | 0.032 | +0.012 | 0.517 | +0.012 |
| same 64 cols, α=10 | 0.764 | +0.744 | 0.655 | +0.150 |
| all 3584 cols, α=1 / 1.5 / 2 (p10c-style rank-1 reflection) | 0.027 / 0.032 / 0.038 | +0.007 / +0.012 / +0.017 | | +0.010 / +0.016 / +0.023 |
| same cols, norm-matched random direction | 0.027 | +0.006 | 0.516 | +0.011 |
| same rule (−6 u uᵀ), 64 random columns | 0.022 | +0.001 | 0.506 | +0.001 |
| constant bias −c·u at L15+L19 MLP output, c=0.5 / 1 / 2 | 0.033 / 0.058 / 0.192 | +0.013 / +0.038 / +0.171 | | +0.043 / +0.125 / +0.450 |

## What this says

1. **Removing the "repetition direction" is not the mechanism.** α=1 (projection removal) does nothing; the gain comes from
   the overshoot: the 64 neurons must actively write −5× their old u-component. Dose is steep and non-linear between α=2 and 10.
   This also explains why p10c (α=1.5 on 30 full matrices) and Astra's early "neuron ablations" failed.
2. **Direction matters** (random direction, same columns and norms: ~0) and **columns matter** (same rule on random columns: ~0,
   because random columns have ~4× smaller uᵀw and are not loop-selective).
3. **An ungated steering vector is not enough.** To match p50c's loop effect (+0.093) a constant −u bias needs c ≈ 1.5, which costs
   ≈ +0.3 NLL on productive text, ~6× the collateral of the edit. The edit's selectivity comes from the neurons' own read side:
   they fire 3–5× more inside loops (Q2), so the −u push is applied mostly where it is needed. A rank-1 *weight* edit on the right
   columns is the simple story; a rank-1 *activation* steer is not.
4. p50c and sel_alpha6 give identical numbers: the closed-form from Q1 reproduces the checkpoint exactly.

Caveat: a teacher-forced proxy on 6 texts. Higher loop NLL means "less locked into the repeated continuation", which is what
breaks loops in generation (p50c closes `</think>` in 894/1060 eval samples vs 141/1060), but the proxy cannot show that the
freed model then reasons correctly; the generation evals (Q6) must do that.
