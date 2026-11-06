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
                        correct=grade_value(c["value"], r["answer"]), in_think=c["start"] < th_end))
    out = dict(id=r["id"], ds=r["ds"], gold=r["answer"], n_tok=len(ids), closed="</think>" in text, loops=k0 is not None,
               onset_tok=k0, onset_char=on_char if k0 is not None else None, onset_in_think=(on_char < th_end) if k0 is not None else None,
               answers=ans)
    if k0 is not None:
        cls = "c_none" if not ans else ("a_correct" if ans[-1]["correct"] else "b_wrong")
        unit = _tok.decode(ids[k0:k0 + N])
        nxt = text[on_char:on_char + len(unit) + 200]
        # period of the loop: distance to the second occurrence of the onset 32-gram
        g = ids[k0:k0 + N]; per = None
        for k in range(k0 + 1, len(ids) - N + 1):
            if ids[k:k + N] == g: per = k - k0; break
        out.update(cls=cls, any_correct_before=any(a["correct"] for a in ans), n_ans_before=len(ans),
                   dist_tok=(k0 - ans[-1]["tok_end"]) if ans else None, loop_unit=unit, loop_period=per,
                   loop_label=loop_label(_tok.decode(ids[k0:k0 + per]) if per else nxt),
                   ctx_before=text[max(0, on_char - 700):on_char], ctx_after=text[on_char:on_char + 500])
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
