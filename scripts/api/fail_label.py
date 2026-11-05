"""LLM failure-type labels for p50c / base outputs (see notes/api_failure_labels.md for sampling, text scheme and rubric).
    python scripts/api/fail_label.py sample                 -> notes/api_fail_sample.json (fixed sample; refuses to overwrite)
    python scripts/api/fail_label.py label <model> [subset] -> notes/api_fail_labels_<model>[_<subset>].json
subset: 'pilot' (10 p50c + 10 base thinking-off), 'kappa50' (fixed 50 stratified items), or 'all'.
"""
import os, sys, json, random, re, collections, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/scratch/yang0531/blf_data"
SETS = {  # name: (file, thinking, rule)
    "p50c_think": (f"{D}/sidefx/screen/screen_p50c_n200x1.jsonl", "on", "all"),
    "base_nothink": (f"{D}/nothink_runpod/base_nothink_runpod.jsonl", "off", "allfail+60c"),
    "p50c_nothink": (f"{D}/nothink_runpod/p50c_nothink_runpod.jsonl", "off", "100f+40c"),
    "base_think": (f"{D}/sidefx/screen/screen_base_n100x1.jsonl", "on", "60"),
    **{f"{a}_think": (f"{D}/sidefx/screen/screen_{a}_n200x{2 if a in ('p50chook', 'loopsel') else 1}.jsonl", "on", "60")
       for a in ["p50chook", "clampp5", "loopsel", "clampmean", "a3"]}}
SAMPLE_F = f"{REPO}/notes/api_fail_sample.json"
CODES = ["correct", "reasoning_error", "over_doubt_wording", "endless_recheck", "paraphrase_cycling", "verbatim_loop", "degenerate",
         "truncated_progress", "other"]


def load(f): return [json.loads(l) for l in open(f)]


def make_sample():
    assert not os.path.exists(SAMPLE_F), "sample exists; not overwriting"
    out = []
    for name, (f, think, rule) in SETS.items():
        R = [r for r in load(f) if r.get("k", 0) == 0]; rng = random.Random(f"{name}-0")
        F = [r for r in R if not r["correct"]]; C = [r for r in R if r["correct"]]
        if rule == "all": pick = R
        elif rule == "allfail+60c": pick = F + rng.sample(C, 60)
        elif rule == "100f+40c": pick = rng.sample(F, 100) + rng.sample(C, 40)
        else: pick = rng.sample(R, int(rule))
        for r in pick:
            out.append(dict(uid=f"{name}|{r['set']}|{r['id']}|{r['k']}", arm=name, think=think, set=r["set"], id=r["id"], k=r["k"],
                            correct=r["correct"], cut=r["cut"], n_tokens=r["n_tokens"], cls=r.get("cls"), loop_end=r.get("loop_end"),
                            stratum=rule if rule in ("all",) or rule.isdigit() else ("fail" if not r["correct"] else "correct"),
                            n_fail_pop=len(F), n_corr_pop=len(C)))
    json.dump(out, open(SAMPLE_F, "w"), indent=0); print(collections.Counter(x["arm"] for x in out), len(out))


def questions():
    q = {}
    for l in open(f"{REPO}/data/gsm8k_test.jsonl"): r = json.loads(l); q[r["id"]] = (r["question"], None)
    for l in open(f"{REPO}/data/math500.jsonl"): r = json.loads(l); q[r["id"]] = (r["problem"], r["answer"])
    return q


def view(t, budget=36000, head=12000, tail=12000, nmid=4, mid=3000):
    t = re.sub(r"\s{200,}", lambda m: m.group(0)[:40] + f"⟨{len(m.group(0)) - 40} whitespace characters omitted⟩", t)
    if len(t) <= budget: return t, False
    a, b = head, len(t) - tail; span = b - a; parts = [t[:head]]; last = head
    for j in range(1, nmid + 1):
        c = a + span * j // (nmid + 1); s = max(c - mid // 2, last); e = min(s + mid, b)
        parts.append(f"\n⟨… {s - last} characters omitted …⟩\n"); parts.append(t[s:e]); last = e
    parts.append(f"\n⟨… {b - last} characters omitted …⟩\n"); parts.append(t[b:])
    return "".join(parts), True


RUBRIC = """You are labelling one output of a small language model (Qwen3.5-0.8B) solving a math problem. Read it and fill in the rubric.

Codes for main_type (the dominant reason the output is not a correct finished solution):
- correct: the grader says correct (ALWAYS use this when the grader says correct).
- reasoning_error: finished (or nearly) with a wrong answer because of an arithmetic / algebra / logic mistake or a single plain misreading, WITHOUT extended doubting of the problem statement.
- over_doubt_wording: the model repeatedly questions or re-interprets WHAT THE PROBLEM ASKS (ambiguity, "maybe it means...", "is this a trick / distractor?", "maybe there is a typo"), and this leads to a wrong answer or keeps it from finishing.
- endless_recheck: the model already has the CORRECT answer and keeps re-verifying the computation without finishing.
- paraphrase_cycling: the same content / argument is restated over and over IN NEW WORDS with no progress (not verbatim, not mainly about the wording of the problem).
- verbatim_loop: the same passage repeats (near-)verbatim until the cap.
- degenerate: whitespace / runaway indentation / garbage tokens dominate the end.
- truncated_progress: cut at the token cap while still making genuine forward progress on a long solution.
- other.

Fields:
- final_answer_reached (bool): the output commits to a definitive final answer (boxed, or an explicit "final answer is ..." that ends the solution). Hitting the cap before committing -> false.
- main_type (one code above).
- also_present (list of codes): other failure patterns that clearly occur in the output (not the main one; may be empty).
- wording_doubt_present (bool): the model questions / re-interprets the wording of the problem at least twice anywhere.
- first_answer (string or null): the FIRST value the model states as its answer to the question (not an intermediate quantity).
- first_answer_status: "correct" / "wrong" (compare with the gold answer) / "none" (never states one) / "not_visible" (probably in an omitted part).
- doubt_after_first: "yes" if after the first stated answer the model questions it or the problem interpretation and re-opens the work (beyond one brief confirmation); "no" if it confirms briefly and wraps up; "na" if there is no first answer.
- doubt_target: "wording" / "computation" / "both" / "none".
- evidence: a quote of at most 30 words from the output supporting main_type.

Long outputs are shown as head + evenly spaced middle chunks + tail, with omitted spans marked; long whitespace runs are collapsed and marked."""


def prompt(it, text, q):
    prob, ans = q[it["id"]]
    gold = it.get("gold")
    return (f"{RUBRIC}\n\n=== PROBLEM ({'GSM8K' if it['set'] == 'gsm8k' else 'MATH-500'}) ===\n{prob}\n\n=== GOLD ANSWER ===\n{gold}\n\n"
            f"=== META ===\nthinking mode: {it['think']}; grader verdict: {'correct' if it['correct'] else 'incorrect'}; "
            f"stopped at the 16k-token cap: {'yes' if it['cut'] else 'no'}\n\n=== MODEL OUTPUT ===\n{text}\n=== END OF OUTPUT ===\n\n"
            "Answer with ONE JSON object with exactly the keys final_answer_reached, main_type, also_present, wording_doubt_present, "
            "first_answer, first_answer_status, doubt_after_first, doubt_target, evidence. No other text.")


def _fieldwise(s):
    """fallback for malformed JSON (seen: broken quoting in the last 'evidence' field): extract each field by regex."""
    j = {}
    for k in ["final_answer_reached", "wording_doubt_present"]:
        m = re.search(rf'"{k}"\s*:\s*(true|false)', s); j[k] = (m.group(1) == "true") if m else None
    for k in ["main_type", "first_answer_status", "doubt_after_first", "doubt_target"]:
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', s); j[k] = m.group(1) if m else None
    m = re.search(r'"first_answer"\s*:\s*(null|"((?:[^"\\]|\\.)*)")', s); j["first_answer"] = m.group(2) if m and m.group(2) is not None else None
    m = re.search(r'"also_present"\s*:\s*\[([^\]]*)\]', s); j["also_present"] = re.findall(r'"([a-z_]+)"', m.group(1)) if m else []
    m = re.search(r'"evidence"\s*:\s*"(.*)', s, re.S); j["evidence"] = m.group(1)[:300] if m else ""
    j["_fieldwise"] = True
    return j


def parse(o):
    if not o: return None
    m = re.search(r"\{.*\}", o, re.S)
    s = m.group(0) if m else o
    try: j = json.loads(s)
    except Exception: j = _fieldwise(s)
    if j.get("main_type") not in CODES: return None
    j["also_present"] = [c for c in (j.get("also_present") or []) if c in CODES and c != j["main_type"]]
    return j


def subset(S, which):
    if which == "all": return S
    rng = random.Random(f"{which}-0")
    if which == "pilot":
        return rng.sample([x for x in S if x["arm"] == "p50c_think"], 10) + rng.sample([x for x in S if x["arm"] == "base_nothink"], 10)
    if which == "kappa50":   # 20 p50c think, 15 base nothink, 5 p50c nothink, 5 base think, 5 other arms
        out = []
        for a, k in [("p50c_think", 20), ("base_nothink", 15), ("p50c_nothink", 5), ("base_think", 5)]:
            out += rng.sample([x for x in S if x["arm"] == a], k)
        return out + rng.sample([x for x in S if x["arm"] in ("p50chook_think", "clampp5_think", "loopsel_think", "clampmean_think", "a3_think")], 5)
    raise ValueError(which)


def run_label(model, which):
    import apicache
    S = json.load(open(SAMPLE_F)); items = subset(S, which); q = questions()
    texts = {}
    for name, (f, _, _) in SETS.items():
        need = {(x["set"], x["id"], x["k"]) for x in items if x["arm"] == name}
        if need:
            for r in load(f):
                if (r["set"], r["id"], r.get("k", 0)) in need: texts[(name, r["set"], r["id"], r.get("k", 0))] = r
    ps = []
    for it in items:
        r = texts[(it["arm"], it["set"], it["id"], it["k"])]; it["gold"] = r["gold"]; v, trunc = view(r["text"]); it["view_truncated"] = trunc
        ps.append(prompt(it, v, q))
    with cf.ThreadPoolExecutor(int(os.environ.get("API_THREADS", 12))) as ex:
        outs = list(ex.map(lambda p: apicache.chat(model, p, 8000), ps))
    res = []
    for it, o in zip(items, outs):
        j = parse(o); res.append(dict(it, label=j, raw=None if j else o))
    nbad = sum(r["label"] is None for r in res)
    fn = f"{REPO}/notes/api_fail_labels_{model}_{which}.json"
    assert not os.path.exists(fn), f"{fn} exists; not overwriting"
    json.dump(dict(model=model, subset=which, n=len(res), parse_fail=nbad, api=apicache.STATS, items=res), open(fn, "w"), indent=0)
    print(f"wrote {fn}; parse fail {nbad}/{len(res)}; {apicache.STATS}")


if __name__ == "__main__":
    if sys.argv[1] == "sample": make_sample()
    else: run_label(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all")
