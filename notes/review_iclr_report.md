# Review notes on "What Did Astra Change?" (Overleaf a798fae, 30 pp: ~10 main + refs + 18 appendix)

The claim-by-claim order that follows the blog's experiments works well and should stay. These notes are about readability for
someone reading the report for the first time. Line numbers refer to `iclr_body.tex` at a798fae.

## 1. Section headings: state the finding

Keep the claim structure, but let each heading say what we found, with the claim/experiment as a prefix. A reader who only
skims the headings should get the whole story.

| Now | Suggested |
|---|---|
| Claim I (Experiment 8): more than early stopping | Claim I (Exp. 8): a stopping rule recovers most of the gain |
| Behavioural controls | The model can solve the problems; it cannot stop |
| From the edit to later attention | The weights change in the MLPs; the effect travels through attention |
| Claim II (Experiment 7): attention-based localisation | Claim II (Exp. 7): the attention difference is real, its cause is elsewhere |
| Where the difference is made | Copying head 10.2 changes nothing; copying the MLPs closes the gap |
| Claim III (Experiment 8): subtracting a repetition direction | Claim III (Exp. 8): the edit flips the direction instead of removing it |
| Overshoot and failed repairs | The push overshoots, and fix and damage share one knob |
| Qwen: a useful direction is not enough | Qwen: cheap scores can reward the worse edit |
| Gemma: convincing readings without an accuracy gain | (fine as is) |

## 2. Qualifiers: one short statement of evidence strength per result, details to the appendix

All of the caveats are correct, but in the main text they interrupt the argument. Suggested rule: each result gets at most one
phrase about how strong it is ("held out", "weak evidence", "one seed"), and the reasons go to *Protocol changes, display
choices and metric checks*.

- The point that steering reuses u and is "a related-intervention check, not independent evidence" is made four times
  (L111–112, L293–296, and the Figure 1 caption). Say it once, in Claim III's tool paragraph.
- Figure 2 caption: "These are summary-based bounds, not raw-data bootstrap intervals" belongs in the appendix.
- L85–87: the 95% normal-approximation intervals can become "no evidence of a gain once thinking is off (CIs in Appendix)".
- L79–82: the explanation of strict accuracy versus 12.5% × 94% can go to Methods. In the main text, "accuracy also credits
  correct answers in truncated text" is enough.
- L295: "r = 0.98 is not an algebraic identity" can go to the appendix.
- The "Evaluation scope" quote block before Claim I reads as a list of limitations before any result. Move it to the closing
  limitations paragraph.
- Fewer digits: head 10.2's 0.9427 / 0.0347 / 0.0075 / 0.8567 can be 0.94 / 0.03 / 0.01 / 0.86.
- A single sentence on its own as a paragraph (L234, "Setting α=1 would instead remove that component.") should merge into
  the preceding one.

## 3. First-time readers

- **Figure 1** is a table of tool families. A first-time reader most needs the mechanism. Suggest a schematic as Figure 1:
  128 selected neurons in L15/L19 → write pushed hard along −u (α = 6 flips and enlarges the component) → later attention
  (L16–23) copies the earlier token less → loops break, with the overshoot marked. The tool-family table can move to Methods or stay as Table 1.
- **Define the edit once, early.** The intro gives "128 MLP output columns", but Eq. 1 and α = 6 only appear in Claim III.
  One sentence in the intro ("Astra pushes 64 columns per layer along −u with strength α = 6") makes Claims I and II easier
  to follow.
- **Name the edits for readers:** at first mention write "Astra's final 50-pass edit (p50c)", "an earlier 10-pass edit
  (p10c)" and "a 25-pass edit that removes an induction head (p25c)". After that, "the final edit" or "the 10-pass edit"
  reads better than the codes, at least in headings and captions.

## 4. Consistent terms

| Concept | Terms now in use | Suggested |
|---|---|---|
| Ending before the token cap | completion, non-truncation, "finishes", "close thinking" | **completion** (define once); keep "closes thinking" only where `</think>` itself matters |
| Verbatim repetition | loops, verbatim loops, repeats, copying runs, loop episodes | **loop** for the behaviour, **copying** for the attention mechanism |
| The teacher-forced score | loop loss, loop-loss effect, teacher-forced effect, cheap proxy | **loop-loss score** |
| u | edit direction, repetition direction, loop direction | **the edit direction u**; "repetition direction" only when quoting Astra |

## 5. Two of our results missing from the main text

- **The rescued problems were already solvable (Claim I).** Among problems the edited model solves with thinking on, base with
  thinking off fails both samples on only 12% (GSM8K) and 8% (MATH-500), against 8–10% expected from sampling noise. This is
  the per-problem version of the thinking-off control, and the most direct evidence that the edit fixes stopping, not ability.
  One or two sentences after L88.
- **The model knows when it is right, but doubts anyway (Claim III, overshoot).** At first-answer points, resid·u separates
  wrong from right answers (AUC 0.80), better than the model's own doubt decision (0.65). With thinking off, correct answers
  draw about 70× less doubt. The edit pushes hardest at wrong answers. This currently appears in one sentence of the Advice
  section. It is a finding and deserves a short paragraph in the overshoot subsection.

## 6. End: merge Pipeline and Advice into one "Lessons"

The two sections overlap: generation-gated acceptance, random-neuron controls, and separating behaviour, localisation and
modification all appear in both. Suggest one section with 4–5 numbered lessons, ending on the general point: neither the
editor's own explanation nor any single tool is complete, so interpretability of AI-written edits needs **complementary tools
and rigor** (pre-registered predictions, causal tests, failures reported). This matches the closing of our blog.

Also:
- The unfinished class-mean / no-repeat-32 validation (currently in Related work, L396–399) is not related work. Move it to
  the limitations paragraph.
- The last paragraph (scope) can become a short "Limitations" heading so readers can find it.

## Smaller items

- TL;DR: good. Consider one concrete number so it is not abstract: "turning thinking off recovers most of the 8→48% gain".
- L29: "26 of 73 held fully" is good to keep; it signals rigor up front.
- Figure 4 caption (sycophancy) is long; panels c and d could share one sentence.
