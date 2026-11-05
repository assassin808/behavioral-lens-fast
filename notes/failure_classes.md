# Failure classes (rule-based, Astra's first-sample logs, 200 problems per set; script `scripts/failure_classes_from_logs.py`)

| class | GSM8K unedited | p10c | p25c | **p50c** | MATH-500 unedited | p10c | p25c | **p50c** |
|---|---|---|---|---|---|---|---|---|
| correct | 23 | 53 | 48 | **122** | 15 | 33 | 26 | **92** |
| cut: verbatim loop (Astra's rule) | **172** | 116 | 147 | 3 | **179** | 144 | 161 | 4 |
| cut: repetitive in paraphrase | 0 | 19 | 0 | 0 | 1 | 11 | 3 | 9 |
| cut: still reasoning, no loop | 4 | 3 | 2 | 0 | 5 | 4 | 6 | 24 |
| cut: degenerate whitespace | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 11 |
| finished, no `\boxed{}` | 0 | 8 | 1 | 13 | 0 | 8 | 4 | **42** |
| finished, wrong answer | 1 | 1 | 2 | 53 | 0 | 0 | 0 | 18 |

- The unedited model's failure is one thing: verbatim loops (86–90 %). p50c removes them almost entirely (3–4 / 200).
- What replaces them: ordinary wrong answers (the model now reaches the hard problems), and three **new** failure modes that are
  side effects of the edit: (a) finishing without a `\boxed{}` answer — 21 % of MATH-500 — (b) whitespace degeneration until the
  cap (5 %), (c) long non-looping reasoning that still does not finish (12 % on MATH-500).
- (a) fits Q1/Q3: the push along −u also moves the state away from the "wrap up / state the result" pattern in some contexts; the
  edited model's answer-formatting is less reliable. Astra's lenient score (last number) recovers part of it. This should be in the
  report as a cost of the edit, and the screening evals report lenient accuracy as well.
- p10c's main effect is to turn verbatim loops into paraphrased loops (19 on GSM8K) — consistent with it damping, not inverting, u.
- Next (API): an LLM judge for "wrong answer" cases (arithmetic slip vs misread problem vs gave up) and for the no-box cases
  (answer present but unboxed vs no answer at all).
