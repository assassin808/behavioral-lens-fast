# Test 4 — SAE features for u and for the edit

Branch `yang/t2-t4`. Scripts: `scripts/test4_*.py`. Activations / checkpoints under `/scratch/yang0531/blf_data/` (not in git).

## Plan
- Our SAEs: BatchTopK (k = 64, 16 384 latents) on **base-model** `hidden_states[16]` (residual after L15) and `hidden_states[20]`
  (after L19). Training tokens: unedited + p50c Astra generations (GSM8K / MATH-500 / IFEval / MMLU-Pro, prompt + generation,
  ≤ 16k tokens each, run through the *base* model) + WikiText-103 (normal non-chat text). Held-out split by generation id.
- Public SAE: `decoderesearch/qwen-3.5-saes`, `qwen-3.5-0.8b/btk-mat-layer-17-k-100` (Matryoshka BatchTopK, 65k, trained on
  lmsys-chat-1m at `model.layers.17` = `hidden_states[18]`). No public SAE exists at hidden_states 16 or 20; this one sits between
  the two edited layers, so it is used as an independent dictionary.
- Edit effect: run base and p50c on the same token sequences, encode both with the base SAE, compare feature activations at
  loop positions (32-gram rule) and productive positions.
- Labels: auto-interp with claude-sonnet-5-5 (Vector gateway) from top-activating contexts, scored by detection on held-out
  activating vs random contexts; spot-check of ~20 features re-labelled and re-scored by claude-opus-5-5 (agreement reported).

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)
- **P4a.** u is not one SAE feature: max |cos(decoder_i, u)| < 0.5 at hidden_states[16] and [20]. Among the top-5 features by
  cos with +u, ≥ 2 get labels about repetition / looping / re-checking ("wait", "let me check again", repeated lines).
- **P4b.** At loop positions the edit turns *off* loop/repetition features: ≥ half of the 10 most-decreased features (by mean
  activation drop, hidden_states[20]) get repetition / re-checking labels.
- **P4c.** The features the edit turns *on* at loop positions are not the features of productive reasoning: < 20 % of the top-20
  most-increased features are among the top-100 productive-vs-loop features of the base model (matches Q5: flag cleared, state
  not returned to normal).
- **Quality gate.** Held-out FVU < 0.2 and < 10 % dead latents; otherwise results are reported as low-confidence.

## Results

(filled in below after the runs)
