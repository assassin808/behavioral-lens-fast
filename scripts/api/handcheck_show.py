"""Print the fixed 20-item hand-check sample (failures, stratified over arms; seed 'hand-0') for manual reading.
    python scripts/api/handcheck_show.py <start> <end>"""
import json, random, sys, re
import fail_label as F
P = json.load(open(f"{F.REPO}/notes/api_fail_labels_claude-sonnet-5-5_all.json"))["items"]
rng = random.Random("hand-0"); pick = []
for arms, k in [(["p50c_think"], 8), (["base_nothink"], 6), (["p50c_nothink"], 2), (["base_think"], 2),
                (["p50chook_think", "clampp5_think", "loopsel_think", "clampmean_think", "a3_think"], 2)]:
    pick += rng.sample([x for x in P if x["arm"] in arms and not x["correct"]], k)
if __name__ == "__main__":
    q = F.questions(); a, b = int(sys.argv[1]), int(sys.argv[2])
    for i, it in enumerate(pick[a:b], a):
        r = [r for r in F.load(F.SETS[it["arm"]][0]) if r["set"] == it["set"] and r["id"] == it["id"] and r.get("k", 0) == it["k"]][0]
        v, tr = F.view(r["text"])
        print(f"\n######## {i} {it['arm']} {it['id']} gold={r['gold']} pred={r['pred']} cut={it['cut']} cls={it['cls']} ntok={it['n_tokens']} len={len(v)}")
        print("PROBLEM:", q[it["id"]][0][:600])
        w = len(re.findall(r"\bWait\b", r["text"])); print(f"#Wait={w}")
        print("HEAD:", v[:1200]); print("\n[MID]:", v[len(v)//2: len(v)//2 + 1000]); print("\n[TAIL]:", v[-1600:])
