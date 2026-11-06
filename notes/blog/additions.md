# New material for the blog (append to the current version; the existing text stays as is)

Each block is self-contained and paste-ready. The newest blocks go at the bottom. Status: ✅ result in, ⏳ experiment still running.

## ✅ A. Were the rescued problems already solvable? (place after "First, separate solving from finishing")

We can check this problem by problem. Take the problems the edited model solves with thinking on, and ask how often the *unedited* model solves them with thinking off. On 12% of them (GSM8K) and 8% (MATH-500), it fails both of its samples. That is no more than the 8–10% expected from sampling noise alone. The edit rescues problems the model could already solve; it does not create new ones.

## ✅ B. The edit removes the copying, not the doubting

What remains after the loops break? We had a language model read the edited model's failed outputs. Seven in ten still doubt: they re-read the wording of the question, or re-check an answer that is already right. The text is no longer verbatim. But re-checking in other words turns out to be common late in *every* long trace, including the successful ones, so it is not a new kind of loop.

## ✅ C. The model knows when it is right, but doubts anyway

Right after its first stated answer, the model's hidden state separates right answers from wrong ones well, as in larger reasoning models (Zhang et al. 2025, arXiv:2504.05419). The projection on *u* alone does this. Yet with thinking on, the model doubts a correct answer almost as often as a wrong one. With thinking off, the same text draws about 70 times less doubt after a correct answer. Astra's edit pushes hardest exactly where doubt was warranted: at wrong answers.

## ✅ D. Related work

Loops like these are common in small reasoning models, and worse in students distilled from larger teachers (Pipis et al. 2025, arXiv:2512.12895). Once started, they feed themselves (Duan et al. 2026, arXiv:2601.05693). Astra is also not alone in finding a sparse fix. Flipping a single neuron reduces repetition loops in Gemma 4, and sparse edits help in Qwen3.5 too (Lazaridis et al. 2026, arXiv:2606.13705).

## ⏳ E. Can the side effects be trimmed without losing the fix? (32k-token rerun of the tail clamp)

## ⏳ F. Where do loops start: after a correct answer, or at an impasse?
