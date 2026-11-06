"""Render markdown tables from report_{name}.json (Exp B). Usage: expB_tables.py report.json > out.md"""
import json, sys
R = json.load(open(sys.argv[1]))
order = ["base_nothink | ALL", "p50c_nothink | ALL", "base_think_preloop | ALL",
         "p50c_think | correct", "p50c_think | finished: wrong answer", "p50c_think | finished: no \\boxed{}",
         "p50c_think | cut: degenerate whitespace", "p50c_think | cut: still reasoning",
         "p50c_think | cut: repetitive (paraphrase)", "p50c_think | cut: verbatim loop", "p50c_think | ALL"]
f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
for key, blk in R.items():
    print(f"\n### {key}\n")
    print("| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |")
    print("|---|---|---|---|---|---|---|")
    for k in order:
        t = blk["table"].get(k)
        if not t: continue
        print(f"| {k} | {t['n']} ({t['n_elig']}) | {f(t['pr_rate'])} | {f(t['frac_cycling'])} | {f(t['pr_rate_pos'])} | {f(t['pr_per_1k'])} | {f(t['vb_rate'])} |")
    for c, v in blk["P2_tests"].items():
        print(f"- still-reasoning minus {c}: ratio {v['ratio']:.2f}, diff 95% CI [{v['diff_ci'][0]:.3f}, {v['diff_ci'][1]:.3f}]")
    if blk["P3"]: print(f"- P3 p50c_think ALL vs base pre-loop: ratio {blk['P3']['ratio']:.2f}, diff CI [{blk['P3']['diff_ci'][0]:.3f}, {blk['P3']['diff_ci'][1]:.3f}]")
