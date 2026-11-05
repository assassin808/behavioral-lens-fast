"""Experiment A shared harness: candidate-answer detection, grading, measurement-prefix construction, token sets.
Used identically for every model / arm (base 0.8B, p50c, Qwen3.5-2B/4B)."""
import glob, gzip, json, os, re, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MATHVERIFY_PKGS", "/scratch/yang0531/blf_models/pylib_sidefx")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from astra_eval import is_correct  # noqa: E402  (Astra's math-verify equivalence)

INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
DSETS = ("gsm8k", "math500")

def questions():
    q = {}
    for l in open(f"{REPO}/data/gsm8k_test.jsonl"):
        r = json.loads(l); q[r["id"]] = r["question"]
    for l in open(f"{REPO}/data/math500.jsonl"):
        r = json.loads(l); q[r["id"]] = r["problem"]
    return q

def user_turn(problem):
    return problem + "\n" + INSTR

def astra_rows(tag, ds):
    rows = []
    for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/{tag}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
    for r in rows: r["ds"] = ds
    return rows

# ---------------- candidate-answer detection
def _box_content(t, i):
    """t[i:] starts with '\\boxed{' -> (content, end index after closing brace) or (None, None)."""
    j = i + len("\\boxed{"); depth = 1
    for k in range(j, len(t)):
        if t[k] == "{": depth += 1
        elif t[k] == "}":
            depth -= 1
            if depth == 0: return t[j:k].strip(), k + 1
    return None, None

_PHRASE = re.compile(r"(?:[Tt]he (?:final )?answer is|[Ff]inal [Aa]nswer(?: is)?\s*[:=]?|\b[Aa]nswer\s*[:=])[ \t]*\**[ \t]*")

def _clean(v):
    v = v.strip().strip("*").strip()
    v = re.sub(r"^\$+|\$+$", "", v.strip()).strip()
    v = v.rstrip(".").rstrip(",").strip().strip("*").strip("$").strip()
    return v

def candidates(think, max_n=3):
    """candidate-answer statements in the thinking text: list of dicts(start, end, value, kind), sorted by start.
    end = index just past the statement (closing brace / end of value)."""
    out = []
    for m in re.finditer(r"\\boxed\{", think):
        c, e = _box_content(think, m.start())
        if c: out.append(dict(start=m.start(), end=e, value=c, kind="boxed"))
    for m in _PHRASE.finditer(think):
        line_end = think.find("\n", m.end()); line_end = len(think) if line_end < 0 else line_end
        rest = think[m.end():line_end]
        if "\\boxed" in rest:  # handled by the boxed rule
            continue
        ls = think.rfind("\n", 0, m.start()) + 1
        before = think[ls:m.start()]
        if re.search(r"\b(if|whether|scenario|assume|suppose|unless|could|would|might|maybe|recall|similar|typical|usually|expected|then|example)\b|\?", before, re.I) or "?" in rest:
            continue   # hypothetical / question, not a statement of an answer
        if re.match(r"\s*not\b", rest): continue
        v = _clean(rest)
        if v and len(v) <= 80 and re.search(r"[0-9]|\\[a-z]", v) and not re.search(r"\b(in|within|inside)\b", v):
            out.append(dict(start=m.start(), end=m.end() + len(rest.rstrip()), value=v, kind="phrase"))
    out.sort(key=lambda d: d["start"])
    res, last_line = [], -1
    for d in out:   # one candidate per line
        ln = think.count("\n", 0, d["start"])
        if ln == last_line: continue
        last_line = ln; res.append(d)
        if len(res) >= max_n: break
    return res

def grade_value(v, gold):
    if is_correct(v, gold): return True
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", v)
    return bool(nums and is_correct(nums[-1].replace(",", ""), gold))

_MARKUP = re.compile(r"[ \t]*(?:(?:[*\-+•]|\d+\.|\*\*|#+|>)[ \t]*)*")

def measure_prefix(think, cand):
    """prefix of the thinking text at which the next token is measured: through the end of the candidate's line, the
    newline(s), and the next line's leading whitespace/list markup, all as in the real trace. None if the trace ends."""
    le = think.find("\n", cand["end"])
    if le < 0: return None
    k = le
    while k < len(think) and think[k] in " \t\n": k += 1   # skip blank / whitespace-only lines
    m = _MARKUP.match(think, k)
    k2 = m.end()
    if k2 >= len(think): return None
    return think[:k2]

_RECHECK = re.compile(r"wait|double[- ]check|verify|re-?check|re-read|check|hmm|actually|\bbut\b", re.I)
def real_recheck(think, cand):
    le = think.find("\n", cand["end"]); le = cand["end"] if le < 0 else le
    return bool(_RECHECK.search(think[le:le + 300]))

# ---------------- token sets (pre-registered in notes/expA_calibrated_doubt.md)
DOUBT = {"wait", "but", "hmm", "actually", "however", "maybe", "perhaps", "double", "verify", "check", "hold",
         "alternatively", "re", "is", "did", "no"}
WRAP = {"so", "therefore", "thus", "final", "</think>", "**", "\\boxed", "answer", "done", "conclusion", "okay", "ok"}
AMBIG = {"let"}
# Amendment 2 (from the pilot, before the full run): secondary, broader sets. Primary analysis keeps DOUBT / WRAP.
DOUBT_EXT = DOUBT | {"if", "could", "what", "why", "which", "alternative", "does", "another", "or"}
WRAP_EXT = WRAP | {"ref", "review", "refining", "formatting", "format", "looks", "ready", "everything", "construct", "draft",
                   "structure", "steps", "step", "writing", "box", "seems"}

def token_sets(tok):
    import numpy as np
    V = len(tok); d, w, a, de, we = (np.zeros(V, bool) for _ in range(5))
    for i in range(V):
        s = tok.decode([i])
        if s and set(s) <= {"\n"} and len(s) >= 2: w[i] = we[i] = True; continue   # '\n\n'-type tokens
        t = s.strip().lower()
        if t in DOUBT: d[i] = True
        elif t in WRAP: w[i] = True
        elif t in AMBIG: a[i] = True
        if t in DOUBT_EXT: de[i] = True
        elif t in WRAP_EXT: we[i] = True
    return d, w, a, de, we

# Amendment 3 (after reading 6 incorrect base points from the full run; applied in analysis to every arm alike):
# a phrase "candidate" whose value is a description rather than a value ("the coefficient of $x^0", "simply the number of integer
# points ... such that ...") is not a candidate answer. Hedge words in front of the value are stripped first.
_HEDGE = re.compile(r"^(?:(?:seems|seem|to|be|is|definitely|clearly|just|simply|probably|likely|indeed|still|consistently|now)\s+)+", re.I)
def answer_like(value):
    v = _HEDGE.sub("", value.strip())
    words = re.findall(r"[A-Za-z]{2,}", re.sub(r"\$[^$]*\$?|\\[a-zA-Z]+(\{[^}]*\})?", " ", v))
    return len(words) <= 2
