# Test 6 (teacher-forced) — α sweep, re-ranked top-64, 20 random directions, random columns

Script `scripts/q4b_tf_sweep.py`, log `notes/q4b_tf_sweep.log`. Same proxy as Q4: NLL of the actual next token at loop positions
(base 0.021) and productive positions (base 0.504) on 6 looping unedited GSM8K generations, 4096 tokens.

| variant (rule w' = w − α·d(dᵀw) on 64 columns) | Δ loop NLL | Δ prod NLL |
|---|---|---|
| Astra cols, d = u, α = 1 / 2 / 4 / **6** / 8 | +0.005 / +0.011 / +0.034 / **+0.093** / +0.312 | +0.005 / +0.012 / +0.029 / +0.053 / +0.089 |
| our top-64 by d′ / by |uᵀw| / combined, d = u, α = 6 | +0.031 / +0.041 / +0.054 | +0.027 / +0.033 / +0.038 |
| Astra cols, 20 random unit directions, α = 6 | mean +0.0001, max +0.0004 | ≈ 0 |
| 5 random column sets, d = u, α = 6 | ≤ +0.001 | ≤ +0.002 |

- **Direction**: 20/20 random directions do nothing (the largest is 0.4 % of Astra's effect). u is specific.
- **Dose**: monotone and convex in α; α = 8 triples the loop effect of α = 6 but also doubles the collateral. Astra's α = 6 sits
  just below the knee. (Generation at α = 8 is in the screening queue to see whether it over-shoots.)
- **Columns**: Astra's 64 columns beat all three of our re-rankings at the same α, even though they are index-skewed
  (median index 364 / 354). So Astra's selection criterion (activation contrast between productive and repetitive contexts,
  measured in long contexts) captures something our 4096-token proxies (d′ of |act|, |uᵀw|) do not. Our rankings are not the
  "true" top-64; a better criterion would be each neuron's actual signed write along u in loops minus productive, in full-length
  contexts. Index skew and quality are therefore not in conflict here: Astra likely scanned low indices first but validated by effect.
