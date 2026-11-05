"""Cached OpenAI-compatible client for the Vector gateway. Cache: sha256(model, messages, max_tokens) -> response text,
one JSON file per request under $API_CACHE (default /scratch/yang0531/blf_data/api_cache). The key is read from the environment
(VECTOR_API_KEY, VECTOR_BASE_URL from ~/.env) and never written anywhere."""
import os, json, hashlib, time, threading
CACHE = os.environ.get("API_CACHE", "/scratch/yang0531/blf_data/api_cache")
os.makedirs(CACHE, exist_ok=True)
_lock = threading.Lock(); STATS = dict(hit=0, miss=0, fail=0)
_client = None
def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=os.environ.get("VECTOR_BASE_URL", "https://proxy.vectorinstitute.ai/v1"),
                         api_key=os.environ["VECTOR_API_KEY"], max_retries=4, timeout=300)
    return _client
def key(model, messages, max_tokens, extra=None):
    return hashlib.sha256(json.dumps([model, messages, max_tokens, extra or {}], sort_keys=True).encode()).hexdigest()
def chat(model, prompt, max_tokens=4000, extra=None, tries=4):
    msgs = [{"role": "user", "content": prompt}]
    k = key(model, msgs, max_tokens, extra); f = f"{CACHE}/{k[:2]}/{k}.json"
    if os.path.exists(f):
        with _lock: STATS["hit"] += 1
        return json.load(open(f))["text"]
    for t in range(tries):
        try:
            kw = dict(model=model, messages=msgs)
            if model.startswith("gpt-5") or model.startswith("gpt-6"): kw["max_completion_tokens"] = max_tokens
            else: kw["max_tokens"] = max_tokens
            kw.update(extra or {})
            r = client().chat.completions.create(**kw)
            txt = r.choices[0].message.content or ""
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(dict(model=model, prompt=prompt, max_tokens=max_tokens, extra=extra, text=txt,
                           usage=r.usage.model_dump() if r.usage else None, finish=r.choices[0].finish_reason), open(f, "w"))
            with _lock: STATS["miss"] += 1
            return txt
        except Exception as e:
            print("api error", model, type(e).__name__, str(e)[:200], flush=True); time.sleep(3 * (t + 1))
    with _lock: STATS["fail"] += 1
    return None
