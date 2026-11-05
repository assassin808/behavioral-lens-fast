"""Re-summarize notes/screen/*.jsonl (strict + lenient accuracy, completion, loop-end, no-box) and print a table."""
import glob, json, os, sys
sys.path.insert(0, "scripts")
from astra_eval import grade_lenient
rows = []
for f in sorted(glob.glob("notes/screen/*.jsonl")):
    if f.endswith("summary.jsonl"): continue
    R = [json.loads(l) for l in open(f)]
    for x in R:
        if "correct_lenient" not in x: x["correct_lenient"] = grade_lenient(x["text"], x["gold"])
    for name in sorted({x["set"] for x in R}):
        S = [x for x in R if x["set"] == name]; done = [x for x in S if not x["cut"]]
        rows.append(dict(tag=os.path.basename(f)[:-6], set=name, n=len(S), completion=len(done)/len(S), acc=sum(x["correct"] for x in S)/len(S),
                         acc_lenient=sum(x["correct_lenient"] for x in S)/len(S), acc_on_completed=sum(x["correct"] for x in done)/max(1,len(done)),
                         loop_end=sum(x["loop_end"] for x in S)/len(S), no_box=sum(1 for x in done if x["pred"] is None)/len(S), mean_tokens=sum(x["n_tokens"] for x in S)/len(S)))
print("tag | set | n | completion | acc | acc_lenient | acc_on_completed | loop_end | no_box_finished | mean_tokens")
for r in rows: print(f"{r['tag']} | {r['set']} | {r['n']} | {r['completion']:.2f} | {r['acc']:.3f} | {r['acc_lenient']:.3f} | {r['acc_on_completed']:.3f} | {r['loop_end']:.2f} | {r['no_box']:.2f} | {r['mean_tokens']:.0f}")
json.dump(rows, open("notes/screen/table.json", "w"), indent=1)
