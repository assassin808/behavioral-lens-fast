"""Amendment 5: a harder held-out set so the 4B has enough wrong first candidates. MATH test (EleutherAI/hendrycks_math parquet),
Level 5, excluding MATH-500 problems (exact problem-text match), gold = last \\boxed of the reference solution; 400 sampled, seed 0.
    python scripts/expA_make_mathhard.py PARQUET_DIR  -> data/expA_mathhard.jsonl"""
import glob, json, os, random, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import REPO
sys.path.insert(0, f"{REPO}/astra_logs/exp8_math/code")
from mathqa import gold_of_solution
m500 = {json.loads(l)["problem"].strip() for l in open(f"{REPO}/data/math500.jsonl")}
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*/test-*.parquet")):
    df = pd.read_parquet(f); sub = os.path.basename(os.path.dirname(f))
    for i, r in df.iterrows():
        if r["level"] != "Level 5" or r["problem"].strip() in m500: continue
        g = gold_of_solution(r["solution"])
        if g: rows.append(dict(id=f"mathhard/{sub}/{i}", problem=r["problem"], answer=g, subject=sub))
random.Random(0).shuffle(rows); rows = rows[:400]
with open(f"{REPO}/data/expA_mathhard.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), {s: sum(r["subject"] == s for r in rows) for s in sorted({r["subject"] for r in rows})})
