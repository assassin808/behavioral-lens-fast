"""Shim for weightenv.lmo.tasks.mcqa.loop_signals (not in the public logs); the rule as described in mathqa.loop_signals' docstring:
in the last 200 words, the 4-word positions whose phrase occurs 3+ times there are at least half of all positions."""
from collections import Counter
def loop_signals(text):
    w = text.split()[-200:]
    grams = [tuple(w[i:i+4]) for i in range(max(0, len(w) - 3))]
    if not grams: return {"repetitive_end": False}
    c = Counter(grams)
    return {"repetitive_end": sum(1 for g in grams if c[g] >= 3) >= 0.5 * len(grams)}
