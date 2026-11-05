"""Rule-based failure classes on Astra's first-sample logs (before any LLM judge): per run and set."""
import glob, gzip, json, sys, re
sys.path.insert(0, "scripts")
from astra_eval import grade, loop_signals, extract
def classify(text, answer):
    ok, pred = grade(text, answer)
    if ok: return "correct"
    closed = "</think>" in text
    tail = text[-2000:]
    if not closed:
        ls = loop_signals(text)
        if re.fullmatch(r"\s*", tail): return "cut: degenerate whitespace"
        if ls["verbatim_loop_end"]: return "cut: verbatim loop"
        if ls["repetitive_end"]: return "cut: repetitive (paraphrase)"
        return "cut: still reasoning"
    if pred is None: return "finished: no \\boxed{}"
    return "finished: wrong answer"
for ds in ["gsm8k", "math500"]:
    print(f"\n## {ds}")
    allc = {}
    for run in ["unedited", "p10c", "p25c", "p50c"]:
        c = {}
        for f in sorted(glob.glob(f"astra_logs/exp8_math/eval/{run}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
            for l in gzip.open(f, "rt"):
                r = json.loads(l); k = classify(r["text"], r["answer"]); c[k] = c.get(k, 0) + 1
        allc[run] = c
    keys = sorted({k for c in allc.values() for k in c})
    print("class | " + " | ".join(allc)); 
    for k in keys: print(f"{k} | " + " | ".join(str(allc[r].get(k, 0)) for r in allc))
