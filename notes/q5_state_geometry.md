# Q5 — How do the internal states move?

Script: `scripts/q5_state_geometry.py`. 8 unedited looping GSM8K generations (loop_frac 4–31 %), 4096 tokens.
A logistic probe "loop vs productive position" is fit on **base** states (leave-one-text-out, balanced), then applied to base and
p50c states of the held-out text.

| hidden_states[L] | probe acc (base, held-out) | P(loop) base, loop pos | P(loop) **edit**, loop pos | P(loop) base, prod pos | P(loop) edit, prod pos | d(edit-loop, prod centroid) / d(base-loop, prod centroid) |
|---|---|---|---|---|---|---|
| 12 | 0.91 | 0.85 | 0.85 | 0.04 | 0.04 | 1.00 |
| 15 (before the edit) | 0.91 | 0.85 | 0.85 | 0.04 | 0.04 | 1.00 |
| 16 | 0.90 | 0.84 | **0.27** | 0.04 | 0.01 | 0.99 |
| 18 | 0.91 | 0.85 | 0.19 | 0.04 | 0.01 | 0.97 |
| 19 | 0.91 | 0.85 | 0.18 | 0.04 | 0.01 | 0.99 |
| 20 | 0.91 | 0.84 | 0.11 | 0.04 | 0.00 | 0.98 |
| 22 | 0.90 | 0.83 | 0.11 | 0.04 | 0.01 | 1.02 |
| 24 (final) | 0.89 | 0.81 | **0.08** | 0.04 | 0.00 | 0.94 |

- "Am I in a loop" is linearly readable from the base residual stream from L12 on (acc 0.9), i.e. before the edited layers:
  the model *represents* that it is looping; it just does not act on it. The edited neurons read that representation (Q2).
- One layer after the edit, the loop-position states no longer read as "loop" to a probe trained on the base model (0.85 → 0.27),
  and by the last layer 0.08. Productive positions are pushed slightly further into "not loop" (0.04 → 0.00).
- But the Euclidean distance from loop states to the productive-state centroid does not shrink (ratio ≈ 1.0). The edit does not
  move the state onto the productive manifold; it moves it along the loop-readout direction (which is ≈ u, see Q3b: Δ is pure −u
  after L15). So the picture is "the loop flag is forcibly cleared / inverted", not "the state is returned to a normal reasoning state".
  That matches generation: the edited model stops repeating verbatim but keeps re-checking in paraphrase (Q0).

Open: cos(probe direction, u) directly; PCA picture of the loop manifold for the write-up; the same probe on *edited-model*
generations (does the edited model still enter the loop state and get kicked out each time?).
