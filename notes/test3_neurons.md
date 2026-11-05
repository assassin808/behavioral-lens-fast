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
Base model (Qwen3.5-0.8B @ 2fc0636), teacher-forced on Astra's logged generations and our base thinking-off generations.
**Context length is the full generation: ≤ 32,768 tokens including the prompt.**
c_j = act_j·(uᵀw_j) uses the base weights and u = `shared/u_L*.npy`. Loop positions are 32-grams seen earlier in the text.

| corpus | n contexts | mean length (tokens) | content |
|---|---|---|---|
| loop | 100 | 32,758 | unedited GSM8K/MATH-500 thinking-on samples cut at 32k, loop_frac ≥ 0.05; 2.71 M loop positions |
| normal | 100 | 2,638 | 34 finished thinking-on + 66 finished thinking-off (loop_frac < 0.01) GSM8K/MATH; 253 k positions |
| non-math | 100 | 19,471 | 50 IFEval + 50 MMLU-Pro unedited thinking-on samples |

The tracked set for exemplars and auto-interp is Astra's 128 neurons plus 16 random neurons per layer.
Scripts: `scripts/test3_collect.py`, `test3_analyze.py`, `test3_explain_prep.py`, `test3_explain.py`, `test3_explain_compare.py`.
Runner: `scripts/portable_t38.sh` (Killarney: `scripts/killarney_t38.sh`).
Results:
- `notes/test3a_ranking_32768_killarney.json`
- `notes/test3_exemplars_32768_killarney.md` (top-4 windows for each of the 128 neurons)
- `notes/test3_exemplar_summary_32768_killarney.json`
- `notes/test3_autointerp_32768_claude-sonnet-5-5_killarney.json` (main run)
- `…claude-opus-5-5_sub8_killarney.json` (spot check) and `notes/test3_autointerp_compare_killarney.json`

## 3a. True top-64 by loop − normal Δ of c_j (100 + 100 contexts, ≤ 32k tokens)

| | L15 | L19 |
|---|---|---|
| overlap of Astra's 64 with the true top-64 (rank by mean Δ) | **44** | **36** |
| split-half noise ceiling (top-64 vs top-64, 50 random halvings) | 61.6 (min 60) | 59.3 (min 54) |
| chance overlap | 1.1 | 1.1 |
| overlap if ranked by Welch t instead of mean Δ | 20 | 12 |
| overlap, within loop texts (loop vs non-loop positions) | 41 | 35 |
| overlap if ranked by \|act\| difference, no uᵀw weighting | 28 | 11 |
| Astra rank median / # in top-128 / # with Δ < 0 | 36 / 51 / 8 | 57 / 41 / 8 |
| ΣΔ Astra-64 / ΣΔ true-top-64 | **0.86** | **0.93** |
| mean write along u of the 64, loop vs normal (Astra) | 0.418 vs 0.062 | 0.745 vs 0.099 |
| same for all 3,584 neurons | 0.529 vs 0.122 | 0.825 vs 0.182 |
| true top-64 overlap with xuanqi's 4k re-rankings (d′ / uᵀw / combined) | 25 / 32 / 36 | 15 / 22 / 29 |

- Astra's columns are close to the best possible 64 on this metric. They hold 86 % (L15) and 93 % (L19) of the Δ that the true
  top-64 holds. The 64 Astra columns alone account for 87 % (L15) and about 100 % (L19) of the whole layer's loop − normal increase in write
  along u. In L19 the other 3,520 neurons' increases and decreases cancel.
- **New: the index skew is a property of the model, not of Astra's search.** 92 % (L15) and 91 % (L19) of the *true* top-64 lie
  below index 1792. Mean \|Δ\| by eighth of the index range is 0.00095, 0.00038, … 0.00013 for L15 and 0.0018, 0.0002, … for L19.
  Loop-relevant neurons are concentrated in indices 0–447 of both layers. This corrects the reading in `PLAN_xuanqi.md`
  ("Astra scanned low indices"): the skew is real, but it does not show that Astra searched only part of the layer.
- The neurons are not math-specific. Inside IFEval and MMLU-Pro texts that loop, the Astra-64 write along u rises at loop positions.
  In L15, by +0.28 on 43 IFEval contexts and +0.16 on 38 MMLU-Pro contexts. In L19, by +0.54 and +0.27.

## 3b. Exemplars (top-16 windows per neuron, ≤ 2 per context, 7-token peak-context dedup, 300 contexts)

| group | share of top-16 windows inside a verbatim loop | sources of top windows (gsm8k / math / ifeval / mmlupro) |
|---|---|---|
| Astra L15 (64) | **0.21** | 321 / 214 / 224 / 265 |
| Astra L19 (64) | **0.05** | 257 / 343 / 181 / 243 |
| random (32) | 0.02 | 112 / 225 / 68 / 107 |

The neurons' *strongest* activations are mostly not inside verbatim loops; the loop effect is a shift in the mean (3a), not the
peak. Typical peaks: the model re-reading or restating the prompt (`Prompt: "The third friend pressed on the door[[bell]]…`), and
self-checking passages, in math and non-math text alike.

## 3c. Auto-interp with detection scoring (Sonnet `claude-sonnet-5-5` as explainer and judge, via the Vector gateway)

Protocol (bx `neuron_explain.py`):
- Explain from 8 top windows; score on 8 held-out top windows.
- Held-out check: no shared (context, position) with the explanation set, plus 7-token peak-context dedup.
- Two negative sets: 'matched' (same source and same loop status) and 'non-loop' (same source, outside loops).
- The gateway has no logprobs, so the judge rates every excerpt of a neuron 0–100 in one shuffled batch. Mean AP; base rate 0.5.

| group | n | AP desc vs matched | AP desc vs non-loop | AP shuffled desc (non-loop) | AP segment-only (non-loop) | AP "inside a loop" (non-loop / matched) |
|---|---|---|---|---|---|---|
| Astra L15 | 64 | **0.81** | **0.84** | 0.63 | 0.66 | **0.745** / 0.63 |
| Astra L19 | 64 | **0.89** | **0.90** | 0.62 | 0.66 | 0.61 / 0.60 |
| random | 32 | 0.90 | 0.91 | 0.62 | 0.68 | 0.61 / 0.59 |

What the descriptions say:
- About re-reading, restating, quoting or self-checking: **34/64 for L15**, 15/64 for L19, 4/32 for random neurons.
- Peak placed explicitly inside a verbatim loop: 11/64 for L15, 8/64 for L19.
- Peak placed explicitly outside a loop: 21/64 for L15, 30/64 for L19.
- These counts come from a keyword regex over the 160 descriptions.

Three informative neurons:
- **L15.64** (AP 1.00 / 1.00; even the loop-only description scores 1.00). "Fires on newline and indentation whitespace tokens (and
  bullet asterisks) at line starts within repeated bulleted reasoning passages". All 16 top windows are inside loops. This is the
  neuron for the textbook repeated bullet line.
- **L15.601** (AP 1.00 / 0.93). "Fires on tokens inside quoted problem text that the model is re-reading or restating while checking
  the prompt's wording". This is the dominant L15 type: it fires on *referring back*, not on looping itself.
- **L15.46** (AP 1.00; largest c in the pilot). "Fires on the first sub-word fragment of a rare or split word or name (e.g.
  "Bur"|j…)". A neuron with an unrelated lexical trigger that Astra's edit flips anyway. Several such neurons sit among the 64.

**Opus spot check** (`claude-opus-5-5` re-explains and re-judges 8 Astra neurons per layer plus 4 random per layer = 24 neurons;
`notes/test3_autointerp_compare_killarney.json`):

| group | n | Opus vs Sonnet description: same trigger (Sonnet judge) | AP desc vs matched, Sonnet / Opus | AP desc vs non-loop, Sonnet / Opus | AP shuffled, Sonnet / Opus |
|---|---|---|---|---|---|
| Astra | 16 | 16/16 | 0.78 / 0.80 | 0.83 / 0.82 | 0.59 / 0.65 |
| random | 8 | 7/8 | 0.87 / 0.88 | 0.89 / 0.92 | 0.73 / 0.59 |

The two explainers agree on the trigger and reach the same detection AP within ±0.03.

Bookkeeping:
- A first Opus run with max_tokens = 80 returned empty descriptions for 20 of 24 neurons; Opus thinks inside max_tokens. It is kept
  as `…_INVALID_maxtokens80.json` and is not used.
- Sonnet main run: 0 empty descriptions; 4 of 15,360 judge scores were missing and set to 0.5.

## Outcome vs predictions (predictions above unchanged)

| prediction | outcome |
|---|---|
| P3a-1 L15 ≥ 20 / 64; L19 10–20 | **pass for L15 (44)**; L19 = 36, above the predicted range (it is lower than L15, as predicted) |
| P3a-2 ≥ 40 % of true-top-64 ΣΔ (L15) | **pass** (86 %) |
| P3a-3 split-half ≥ 40 | **pass** (62 / 59) |
| P3b ≥ 70 % of top windows inside loops | **fail** (21 % L15, 5 % L19) |
| P3c-1 AP(desc) − AP(shuffled) ≥ 0.1 vs non-loop | **pass** (+0.21 L15, +0.28 L19) |
| P3c-2 AP vs loop-matched ≤ 0.65 | **fail**, and the kill/alt condition holds (0.81 / 0.89 ≥ 0.75): the neurons carry token-level content beyond the loop flag |

Reading:
- Astra picked neurons whose *average* write along u rises in loops (3a, near-optimal).
- Individually they are ordinary, describable neurons. The L15 set is enriched for "referring back to earlier text" (re-reading
  the prompt, quoting, self-checking), and their peaks are mostly outside verbatim loops.
- So the edit pushes −u whenever the model re-reads or restates, not only in loops. That fits the edited model's flat, low u at all
  positions (test 8). It is also a candidate (hypothesis) for the side effects: "no \boxed{}" and less re-checking.
- Cheap test of that hypothesis: compare the edited model's c_j on re-reading spans vs other non-loop spans.

Caveats:
- Explainer and judge are the same model. Detection on random neurons is as high as on Astra's, so the AP numbers say the
  descriptions are specific, not that these neurons are special.
- The batched 0–100 judging is not comparable 1:1 with bx's logprob P(Yes).
- The MMLU-Pro prompt is an approximation of the official one.
- One sample per problem.
