"""Shared helpers for tests 3 and 8 (branch yang/t3-t8): corpus loading, prompts, loop mask, model paths.

Model dirs (rebuilt locally on Killarney, see notes/test3_neurons.md):
  base  = Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17
  p50c  = base + astra_logs/exp8_math/runs/p50c/edit via astra_logs/exp8_math/code/apply_edit.py (sha256-checked)
"""
import glob, gzip, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KILLARNEY = "/scratch/yang0531/blf_models"           # fallback paths (Killarney); portable runs set the env vars below
MODEL_DIR = dict(base=os.environ.get("BLF_MODEL_BASE", f"{_KILLARNEY}/base"), p50c=os.environ.get("BLF_MODEL_P50C", f"{_KILLARNEY}/p50c"))
EXT = f"{REPO}/data"                                   # data/ext_ifeval.jsonl (google/IFEval), data/ext_mmlupro_used.jsonl (TIGER-Lab/MMLU-Pro)
RUNTAG = os.environ.get("BLF_RUNTAG", "killarney")     # suffix of every committed result file (cluster / run name)


def weights_file(tag):
    import glob as _g
    return sorted(_g.glob(f"{MODEL_DIR[tag]}/model*.safetensors"))[0]
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."   # Astra mathqa.INSTRUCTION
DATA_OUT = os.environ.get("T38_DATA") or (f"{os.environ['BLF_DATA']}/t38" if os.environ.get("BLF_DATA") else f"{REPO}/notes/t38_data")   # large per-sample outputs (gitignored)


def loop_mask(ids, n=32):
    """token i is 'in loop' if the n-gram ending at i already occurred earlier (same rule as scripts/q2_neuron_activations.py)."""
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m


def _questions():
    q = {}
    for l in open(f"{REPO}/data/gsm8k_test.jsonl"):
        r = json.loads(l); q[r["id"]] = r["question"] + "\n" + INSTR
    for l in open(f"{REPO}/data/math500.jsonl"):
        r = json.loads(l); q[r["id"]] = r["problem"] + "\n" + INSTR
    if os.path.exists(f"{EXT}/ext_ifeval.jsonl"):
        for l in open(f"{EXT}/ext_ifeval.jsonl"):
            r = json.loads(l); q[r["id"]] = r["prompt"]
    if os.path.exists(f"{EXT}/ext_mmlupro_used.jsonl"):
        for l in open(f"{EXT}/ext_mmlupro_used.jsonl"):
            r = json.loads(l)
            opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
            # approximation of the official MMLU-Pro CoT prompt (Astra's exact prompt builder is not in the logs)
            q[r["id"]] = (f"The following is a multiple choice question about {r['category']}. Think step by step and then finish your "
                          f"answer with \"the answer is (X)\" where X is the correct letter choice.\n\nQuestion: {r['question']}\nOptions:\n{opts}")
    return q


QUESTIONS = None


def question(id_):
    global QUESTIONS
    if QUESTIONS is None: QUESTIONS = _questions()
    return QUESTIONS.get(id_)


def astra_rows(model_tag, ds):
    """Astra's logged first samples (thinking on, 32k cap). model_tag in {unedited, p50c}; ds in {gsm8k, math500, ifeval, mmlupro1k}."""
    rows = []
    for f in sorted(glob.glob(f"{REPO}/astra_logs/exp8_math/eval/{model_tag}/parts/{ds}_s*/{ds}_first_samples.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
    for r in rows: r["src"] = {"gsm8k": "gsm8k", "math500": "math", "ifeval": "ifeval", "mmlupro1k": "mmlupro"}[ds]
    return rows


def encode(tok, id_, text, think=True, maxt=16384):
    """prompt ids + generation ids (truncated to maxt total); returns ids, P (prompt length) or None if no question."""
    q = question(id_)
    if q is None: return None, None
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
    pid = tok(prompt).input_ids; gid = tok(text, add_special_tokens=False).input_ids
    return (pid + gid)[:maxt], len(pid)
