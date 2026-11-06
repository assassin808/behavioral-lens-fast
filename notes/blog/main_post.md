# What weight-space programming actually programmed

*HaoYang Shang (University of British Columbia) — draft; full author list to be confirmed. The full research notebook follows as an appendix.*

GPT-6 Astra changed 0.016% of Qwen3.5-0.8B's weights, with no training, and the model's MATH-500 score rose from 7.9% to 48.1% [1]. We rebuilt the edited model and followed the change through it with ten interpretability tools, checking each reading with a causal test.

**In short:**

- **The edit fixes finishing, not reasoning.** The small model could already solve these problems. It just could not stop: it copied its own text in loops until the token budget ran out. Turning thinking off, a switch every user has, scores as well. The problems the edit rescues are ones the model could already solve.
- **It works by stopping copying.** 128 neurons that fire when the model re-reads or re-checks now push hard from "doubt" towards "wrap up". That push travels through later attention, which stops copying the earlier text, so loops break.
- **The fix and its side effects are one knob.** The push overshoots normal states and causes new failures, such as long runs of whitespace. Every way we found to weaken it brought the loops back.
- **The model still doubts.** It knows when an answer is right, yet in thinking mode it doubts anyway. The edit removes the copying, not the doubting.
- **Several standard tools gave confident, wrong readings.** Only causal checks caught them.

## 1. The problem: solving vs. finishing

Without the edit, the model finishes only about one GSM8K problem in eight within its budget, but gets almost all of those right. The rest end in loops in which it copies its own earlier text word for word.

Problem by problem, the problems the edited model solves are ones the unedited model already solves with thinking off. The "new" ones are no more frequent than sampling noise. Forcing the model to stop thinking at 4k or 8k tokens recovers most of the gain. With thinking off, the edit adds nothing.

## 2. What Astra changed

The edit flips and amplifies what 128 neurons in two layers write along one direction, *u*. Those neurons mostly fire when the model goes back to re-read the problem or check its work.

Adding *+u* to real hidden states favours *Wait* and *maybe*; adding *−u* favours *Final* and ending the thought. The edit's whole effect on the output matches adding *−u* almost exactly. The precise choice of neurons matters little: other well-ranked sets work as well, while random directions on Astra's neurons do nothing.

## 3. Why it works

The neurons do not write "stop" into the output; their direct path carries 3% of the effect. Instead, the push changes what later attention looks for. Attention to the earlier copy of the text drops by a third, so the model stops copying. Loops still start, but they end after about 70 tokens instead of about 1,800.

## 4. What it costs

The push does not return loop states to normal; it overshoots past them. The edited model shows failures the original never did, such as long runs of whitespace. Weakening the push in three different ways removed most of them, but also brought the loops back. Only trimming the far tail of the overshoot looked promising, and we are testing it properly now.

## 5. Why training did not fix this

Loops like these are common in small reasoning models, and worse in students distilled from larger teachers [2, 3]. Two ordinary training pressures meet here:

- **Pretraining rewards copying.** When text repeats, continuing the repeat is the right prediction, and the unedited model predicts its own looped tokens almost perfectly.
- **Long-thinking training rewards re-checking.** A small model can learn the habit without learning when to stop.

The model's hidden state does know when an answer is right, as in larger reasoning models [4]. Yet in thinking mode it doubts a correct answer almost as often as a wrong one. Outside thinking mode, the same text draws about 70 times less doubt after a correct answer.

Astra's edit cuts the copying, but it pushes hardest exactly where doubt was warranted, at wrong answers. Its failures are still doubt: re-reading the question, or re-checking an answer that is already right.

## 6. Which tools to trust

Most tools pointed the right way. Several gave confident readings that only a causal test exposed:

- **Probe weights.** The weights of a loop probe looked unrelated to *u*, even though *u* alone separates loops.
- **Attention.** Copy attention looked unchanged when measured at one offset, and fell by a third at the right one.
- **Auto-interpretation.** It gave Astra's neurons good labels, but random neurons scored as well, under judges from two model families.
- **Logit lens.** In the middle layers it read the edit backwards.

Activation steering and path patching, which intervene instead of observe, agreed with each other and with the edit.

## 7. What this means

Weight-space programming worked here, but as a patch, not an upgrade. Astra found the right place and the right direction. Humans have recently found similar sparse fixes for loops in other models [5]. What the agent added was the search. What it did not add was an explanation.

We found the same pattern in Astra's sycophancy experiment. Telling the sycophantic model from the honest one is easy. Astra's explanation of where the difference lives fails a causal check: copying the head it named changes no answer, while copying the MLPs closes the whole gap.

Edits written by AI systems will need the same debugging as code: a statement of what was changed, why it works, and what it costs.

*Appendix: the full research notebook, with every number, method and pre-registered prediction.*

[1] Takehi, Teoh, Zhang & Khattab, *Sparks of Weight-Space Programming*, MIT CSAIL, 2026.
[2] Pipis et al., *Wait, Wait, Wait... Why Do Reasoning Models Loop?*, arXiv:2512.12895.
[3] Duan et al., *Circular Reasoning*, arXiv:2601.05693.
[4] Zhang et al., *Reasoning Models Know When They're Right*, arXiv:2504.05419.
[5] Lazaridis et al., *When Can One Neuron Fix Repetition Loops in LLMs?*, arXiv:2606.13705.
