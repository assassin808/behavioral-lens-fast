# Test 3 — activation analysis of Astra's 128 neurons (branch `yang/t3-t8`)

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)

- **P3a-1** With ≥50 looping + ≥50 normal contexts at 16k tokens, the true top-64 by loop−normal difference of
  c_j = act_j·(uᵀw_j) overlaps Astra's L15 columns by **≥ 20 / 64** (xuanqi at 4k, 12 texts: 31 by uᵀw, 25 by d′); L19 lower (**10–20**).
- **P3a-2** Astra's 64 columns capture **≥ 40 %** of the summed Δ of the true top-64 in L15.
- **P3a-3** The split-half overlap of the true top-64 with itself is ≥ 40 (the ranking is stable enough that the overlap with Astra is meaningful).
- **P3b** Top exemplars of Astra's L15 neurons are dominated by repeated text: **≥ 70 %** of top-16 windows lie inside a
  verbatim-repeated passage, and IFEval / MMLU-Pro loops appear among them (not math-specific).
- **P3c-1** Descriptions mostly name repetition / restating; AP(description) > AP(shuffled) by ≥ 0.1 against **non-loop** negatives.
- **P3c-2** Against **loop-matched** negatives (same source, also inside a loop), AP(description) is ≤ 0.65 on average: most of
  what the neurons detect is "being in a loop", not a specific token. Kill/alt: if AP vs matched negatives ≥ 0.75, the
  neurons carry token-level content beyond the loop flag.

## Setup
