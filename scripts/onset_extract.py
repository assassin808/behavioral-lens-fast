"""Loop-onset analysis, stage 1 (CPU): onset, stated answers (graded), loop content, per trace.
Definitions pre-registered in notes/loop_onset.md.

    python scripts/onset_extract.py SOURCE OUT [MAX_TRACES]
SOURCE: fresh (gen_base_p0+p1), astra (Astra unedited logs gsm8k+math500), hard (gen_base_hard).
OUT: jsonl, refuses to overwrite. One row per trace.
"""
import bisect, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expA_common import REPO, DSETS, astra_rows, questions, candidates, grade_value, answer_like

D = os.environ.get("BLF_DATA", "/scratch/yang0531/blf_data")
TOKDIR = "/scratch/yang0531/blf_models/base"
N = 32

def load(source):
    if source == "fresh":
        return [json.loads(l) for f in ("gen_base_p0", "gen_base_p1") for l in open(f"{D}/expA/{f}.jsonl")]
    if source == "hard":
        return [json.loads(l) for l in open(f"{D}/expA/gen_base_hard.jsonl")]
    if source == "astra":
        Q = questions(); rows = []
        for ds in DSETS:
            for r in astra_rows("unedited", ds):
                rows.append(dict(id=r["id"], ds=ds, problem=Q[r["id"]], answer=r["answer"], text=r["text"]))
        return rows
    raise ValueError(source)

def onset_tok(ids):
    cnt, first = {}, {}
    for k in range(len(ids) - N + 1):
        g = tuple(ids[k:k + N]); cnt[g] = cnt.get(g, 0) + 1
        if g not in first: first[g] = k
    starts = [first[g] for g, c in cnt.items() if c >= 3]
    return min(starts) if starts else None

_RECHK = re.compile(r"wait|check|verif|double|hmm|actually|let me re|confirm|make sure|is that right|correct\?", re.I)
_REREAD = re.compile(r"the (problem|question)|\bsays\b|\bstates\b|\basks\b|re-?read|reading|\"[^\"]{6,}\"", re.I)
_COMP = re.compile(r"\d\s*[-+*/×÷=]\s*\d|\d\s*=|=\s*\d|calculat|comput", re.I)
def loop_label(s):
    if _RECHK.search(s): return "recheck"
    if _REREAD.search(s): return "reread"
    if _COMP.search(s): return "computing"
    return "other"

_tok = None
def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOKDIR)

_GC = {}
def grade(v, gold):
    k = (v, gold)
    if k not in _GC: _GC[k] = grade_value(v, gold)
    return _GC[k]

def onset_noquote(ids, problem, W=8):
    """amendment 1 S2-i: as onset_tok, but 32-grams that quote the problem (contain a run of >= 8 words that occurs in the
    whitespace-normalised problem text) are ignored."""
    P = " " + " ".join(problem.split()) + " "
    cnt, first = {}, {}
    for k in range(len(ids) - N + 1):
        g = tuple(ids[k:k + N]); cnt[g] = cnt.get(g, 0) + 1
        if g not in first: first[g] = k
    starts = sorted(first[g] for g, c in cnt.items() if c >= 3)
    for k in starts:
        w = _tok.decode(ids[k:k + N]).replace('"', " ").split()
        quote = any(" " + " ".join(w[i:i + W]) + " " in P.replace('"', " ").replace("  ", " ") for i in range(1, max(1, len(w) - W)))
        if not quote: return k
    return None

def onset_terminal(ids, pre=64, tail=2000):
    """amendment 1 S2-ii: last test-8 run start (preceded by >= pre non-loop tokens) in traces whose last `tail` tokens are >= 50% loop."""
    seen, m = set(), [False] * len(ids)
    for i in range(N, len(ids)):
        g = tuple(ids[i - N:i]); m[i] = g in seen; seen.add(g)
    if len(ids) < tail or sum(m[-tail:]) < tail / 2: return None
    last = None
    for i in range(pre, len(ids)):
        if m[i] and not m[i - 1] and not any(m[i - pre:i]): last = i
    return last - N if last is not None and last >= N else last   # m[i] marks the END of a repeated 32-gram; its copy starts at i-N

_RES = re.compile(r"(?:=|\bis\b)([^=\n]{1,60})$")
def computed_results(text, starts, gold, limit_char):
    """amendment 1 S1: lines ending in '= X' / 'is X' (gold-blind), graded."""
    out, pos = [], 0
    for line in text[:limit_char].split("\n"):
        ln = line.rstrip()
        m = _RES.search(ln)
        if m:
            v = m.group(1).strip().strip("*").strip().rstrip(".").strip().strip("$").strip().rstrip(".").strip()
            if v and len(v) <= 40 and re.search(r"[0-9]|\\[a-z]", v) and answer_like(v):
                e = pos + len(ln)
                out.append(dict(end=e, tok_end=bisect.bisect_left(starts, e), value=v, correct=grade(v, gold)))
        pos += len(line) + 1
    return out

def classify(alist, t):
    """class at token depth t given a list of graded statements with tok_end."""
    prev = [a for a in alist if a["tok_end"] <= t]
    if not prev: return "c_none", None
    return ("a_correct" if prev[-1]["correct"] else "b_wrong"), t - prev[-1]["tok_end"]

def process(r):
    text = r["text"]
    enc = _tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    starts = [a for a, b in offs]
    k0 = onset_tok(ids)
    on_char = offs[k0][0] if k0 is not None else len(text)
    region = text[:on_char]
    th_end = text.find("</think>"); th_end = len(text) if th_end < 0 else th_end
    ans = []
    for c in candidates(region, 10 ** 6):
        if c["end"] > on_char or not answer_like(c["value"]): continue
        tend = bisect.bisect_left(starts, c["end"])            # first token starting at/after the statement end
        ans.append(dict(start=c["start"], end=c["end"], tok_end=tend, value=c["value"], kind=c["kind"],
                        correct=grade(c["value"], r["answer"]), in_think=c["start"] < th_end))
    # whole-text lists (for the null and the S2 onset variants); capped at the first 20k tokens' worth of text for speed
    lim = offs[min(len(offs), 20000) - 1][1] if offs else 0
    ans_all = []
    for c in candidates(text[:lim], 10 ** 6):
        if not answer_like(c["value"]): continue
        ans_all.append(dict(tok_end=bisect.bisect_left(starts, c["end"]), correct=grade(c["value"], r["answer"])))
    res_all = computed_results(text, starts, r["answer"], lim)
    k_nq = onset_noquote(ids, r["problem"]) if k0 is not None else None
    k_term = onset_terminal(ids)
    out = dict(id=r["id"], ds=r["ds"], gold=r["answer"], n_tok=len(ids), closed="</think>" in text, loops=k0 is not None,
               onset_tok=k0, onset_char=on_char if k0 is not None else None, onset_in_think=(on_char < th_end) if k0 is not None else None,
               onset_noquote=k_nq, onset_terminal=k_term, answers=ans, answers_all=ans_all, results_all=res_all)
    if k0 is not None:
        cls = "c_none" if not ans else ("a_correct" if ans[-1]["correct"] else "b_wrong")
        unit = _tok.decode(ids[k0:k0 + N])
        g = ids[k0:k0 + N]; per = None
        for k in range(k0 + 1, len(ids) - N + 1):
            if ids[k:k + N] == g: per = k - k0; break
        out.update(cls=cls, any_correct_before=any(a["correct"] for a in ans), n_ans_before=len(ans),
                   dist_tok=(k0 - ans[-1]["tok_end"]) if ans else None, loop_unit=unit, loop_period=per,
                   loop_label=loop_label(_tok.decode(ids[k0:k0 + per]) if per else text[on_char:on_char + 400]),
                   ctx_before=text[max(0, on_char - 700):on_char], ctx_after=text[on_char:on_char + 500])
    for nm, kk in (("noquote", k_nq), ("terminal", k_term)):
        if kk is not None:
            c = offs[kk][0]; g = ids[kk:kk + N]; per = None
            for k in range(kk + 1, len(ids) - N + 1):
                if ids[k:k + N] == g: per = k - kk; break
            out[f"ctx_{nm}"] = [text[max(0, c - 500):c], text[c:c + 400]]
            out[f"label_{nm}"] = loop_label(_tok.decode(ids[kk:kk + per]) if per else text[c:c + 400])
    return out

if __name__ == "__main__":
    source, of = sys.argv[1], sys.argv[2]
    assert not os.path.exists(of), f"refusing to overwrite {of}"
    rows = load(source)
    if len(sys.argv) > 3: rows = rows[: int(sys.argv[3])]
    with Pool(int(os.environ.get("NPROC", 8)), initializer=_init) as p:
        res = p.map(process, rows, chunksize=4)
    with open(of, "w") as f:
        for x in res: f.write(json.dumps(dict(x, source=source)) + "\n")
    print(f"SUMMARY {of} traces={len(res)} loops={sum(x['loops'] for x in res)}", flush=True)
