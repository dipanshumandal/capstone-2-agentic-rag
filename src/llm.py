"""
Small wrapper around Gemini.
- caches every response on disk (free tier is 5 requests / minute and only 20 / day on the main
  model, re-running the notebook should not burn it again)
- falls back to the lite model when the main one hits the quota
- MOCK_LLM=1 returns an extractive answer from the context, used in CI
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL, GEMINI_FALLBACK_MODEL, MOCK_LLM, CACHE_DIR

_clients = {}
# the 429 retry warning from langchain prints the whole quota message, the fallback handles it anyway
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)


def _client(model):
    if model not in _clients:
        # max_retries=1: on a 429 we would rather switch to the lite model than wait a minute
        _clients[model] = ChatGoogleGenerativeAI(model=model, temperature=0.2, max_retries=1)
    return _clients[model]


def _cache_path(model, prompt):
    key = hashlib.sha256((model + "||" + prompt).encode()).hexdigest()[:24]
    return Path(CACHE_DIR) / f"{key}.json"


def call_llm(prompt, use_cache=True):
    """Returns (text, model_used). model_used is 'cache', 'mock' or the gemini model name."""
    if MOCK_LLM:
        return mock_response(prompt), "mock"

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for model in [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]:
        path = _cache_path(model, prompt)
        if use_cache and path.exists():
            return json.loads(path.read_text())["text"], "cache"

    last_error = None
    for model in [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]:
        try:
            text = _client(model).invoke(prompt).content
            if isinstance(text, list):   # newer versions sometimes return content parts
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
            _cache_path(model, prompt).write_text(json.dumps({"model": model, "text": text}))
            return text, model
        except Exception as e:
            last_error = e
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                print(f"[llm] {model} quota hit, trying fallback")
                continue
            raise

    # both models out of quota -> extractive fallback so the app keeps working
    print(f"[llm] all models failed ({last_error}), using extractive answer")
    return mock_response(prompt), "mock"


def mock_response(prompt):
    # take the first "Support answer:" from the context block of the prompt (or the first
    # policy text) and return it in the json shape the response agent expects
    m = re.search(r"Support answer: (.+?)(?:\n\[|\n---|\Z)", prompt, re.DOTALL)
    if m:
        answer = m.group(1).strip()
    else:
        m = re.search(r"CONTEXT:\n(.+?)\n\n", prompt, re.DOTALL)
        answer = (m.group(1).strip()[:600] if m else "I could not find this in the knowledge base.")
    answer = " ".join(answer.split())[:700]
    return json.dumps({"answer": answer, "sources_used": [1], "confidence": 0.5})


def parse_json(text):
    # gemini sometimes wraps the json in ```json fences or adds a sentence before it
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"answer": text, "sources_used": [], "confidence": 0.4}
