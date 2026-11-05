# Q3 — Who reads what the edit writes? (teacher-forced, base vs p50c on the same looping texts)

Scripts: `scripts/q3_who_reads_u.py` (4 texts, 3072 tokens, eager attention), `scripts/q3b_resid_delta.py` (6 texts, 4096 tokens).

## Size and direction of the perturbation

Relative change of the residual stream, ‖h_edit − h_base‖ / ‖h_base‖ (mean over positions):

| after layer | loop positions | productive positions | cos(Δ, −u15) loop/prod | Δ energy in span(u15,u19) loop/prod |
|---|---|---|---|---|
| L15 | 0.48 | 0.16 | 1.00 / 0.87 | 1.00 / 0.98 |
| L18 | 0.45 | 0.15 | 0.76 / 0.63 | 0.62 / 0.50 |
| L19 | 0.51 | 0.21 | 0.77 / 0.56 | 0.73 / 0.59 |
| L21 | 0.49 | 0.20 | 0.69 / 0.50 | 0.58 / 0.44 |
| L23 (final) | 0.64 | 0.25 | 0.28 / 0.21 | 0.10 / 0.08 |

- The push is ~3× larger inside loops than outside (the gating from Q2), and it is a *large* perturbation: half the residual norm.
  The L15 MLP's own output changes by ~98 % of its norm at loop positions, i.e. the edit roughly replaces that layer's MLP write in loops.
- Right after L15 the change is purely −u. It stays mostly in the u-plane through L21 and is then converted into other directions
  by L22–L23: only 10 % of the final-layer Δ is still along u. So the **direct path (−u → unembedding) is minor**; the logit effect is
  mediated mainly by the last two blocks, consistent with the weak logit-lens picture in Q1.
- Every downstream component (attention and MLP, L16–L23) changes its output by 45–68 % at loop positions; no single reader stands out.

## Copy attention does not explain it

Attention mass from a loop token to the token it is copying (full-attention layers; 8 heads each):

| | exact token | ±1 window |
|---|---|---|
| L19 base → edit | 0.028 → 0.028 | 0.31 → 0.27 |
| L23 base → edit | 0.125 → 0.125 | 0.43 → 0.33 |

The copy heads (L23 head 6: 0.07 exact, 0.06 after; L23 head 2 goes up) barely change. The edit does not "break the copy head";
it changes the state that the heads and MLPs read. This also matches p25c (ablating L15 head 7, Astra's copy-head attempt) gaining
only 14.1 vs 48.1.

## Reading

Mechanism so far: loop-selective neurons in L15/L19 → large −u write at loop positions → the residual state is moved off the
loop manifold → L22–L23 translate the moved state into different next-token distributions (loop NLL 0.02 → 0.11, Q4). It is a
state-level intervention rather than a circuit-level one. Q5 tests the "pushed out of the loop attractor" picture directly.
