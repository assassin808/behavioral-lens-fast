"""Thin wrapper around Astra's public evaluation code (astra_logs/exp8_math/code): same prompt, sampling, grader, loop signals."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "astra_logs", "exp8_math", "code")
os.environ.setdefault("MATHVERIFY_PKGS", "/data2/ec2-user/mathverify_pkgs")
for p in (HERE, CODE):
    if p not in sys.path: sys.path.insert(0, p)
import mathqa as Q  # noqa: E402
SAMPLING_THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}
build_prompt, extract, is_correct, loop_signals = Q.build_prompt, Q.extract, Q.is_correct, Q.loop_signals
def grade(text, answer):
    pred, _ = extract(text); return bool(is_correct(pred, answer)), pred
def three_numbers(samples):
    """samples: list of dict(text, answer, cut). Returns completion rate, accuracy on completed, repetitive-end rate (Astra's rule)."""
    n = len(samples); done = [s for s in samples if not s["cut"]]
    acc_done = sum(grade(s["text"], s["answer"])[0] for s in done) / max(1, len(done))
    rep = sum(loop_signals(s["text"])["repetitive_end"] or loop_signals(s["text"])["verbatim_loop_end"] for s in samples) / n
    return {"n": n, "completion": len(done) / n, "acc_on_completed": acc_done, "acc_all": sum(grade(s["text"], s["answer"])[0] for s in samples) / n, "repetitive_end": rep}
