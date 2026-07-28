"""Nautilus (NRP) managed-LLM access via the OpenAI-compatible Envoy AI Gateway.

Mirrors harness.azure_llm's interface (chat / last_meta / deployments) so the batch
runner can route generation to NRP open-weights models (qwen3, gpt-oss, kimi,
glm-5, minimax-m2, gemma, qwen3-small, ...) with the same envelope logging.

Auth: a bearer token created at https://nrp.ai/llmtoken. Provide it via the
NRP_API_KEY env var or a .secrets/nrp_llm.json file: {"api_key": "..."}.
The endpoint /v1/models is open; /v1/chat/completions requires the token.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = _ROOT / ".secrets" / "nrp_llm.json"
BASE_URL = os.environ.get("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1")

# Open-weights catalog as of provisioning (see /v1/models for the live list).
KNOWN_MODELS = [
    "qwen3", "qwen3-small", "gpt-oss", "kimi", "glm-5", "minimax-m2",
    "gemma", "gemma-small",
]


def _api_key() -> str:
    key = os.environ.get("NRP_API_KEY")
    if key:
        return key.strip()
    if _CFG_PATH.exists():
        cfg = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
        if cfg.get("api_key"):
            return str(cfg["api_key"]).strip()
    raise RuntimeError(
        "No NRP token. Set NRP_API_KEY or create .secrets/nrp_llm.json "
        '{"api_key": "..."} with a token from https://nrp.ai/llmtoken')


_client_cache: dict[str, OpenAI] = {}


def _client() -> OpenAI:
    key = _api_key()
    if _client_cache.get("key") != key:
        _client_cache.clear()
        _client_cache["key"] = key
        _client_cache["client"] = OpenAI(
            base_url=BASE_URL, api_key=key, timeout=240.0, max_retries=0)
    return _client_cache["client"]


def deployments() -> list[str]:
    """Live model ids from the (open) /v1/models endpoint, falling back to the
    known catalog if the listing is unreachable."""
    try:
        models = _client().models.list()
        return [m.id for m in models.data]
    except Exception:  # noqa: BLE001
        return list(KNOWN_MODELS)


def _envelope(resp) -> dict:
    """The metadata the LLM service itself emits -- proof of a real API call
    (request id, served model, token usage, finish reason)."""
    try:
        d = resp.model_dump()
    except Exception:  # noqa: BLE001
        return {}
    ch = (d.get("choices") or [{}])[0]
    return {
        "id": d.get("id"),
        "served_model": d.get("model"),
        "created": d.get("created"),
        "system_fingerprint": d.get("system_fingerprint"),
        "usage": d.get("usage"),
        "finish_reason": ch.get("finish_reason"),
        "provider": "nrp",
    }


_last_meta: dict = {}


def last_meta() -> dict:
    return dict(_last_meta)


def chat(model: str, messages: list[dict], max_tokens: int = 2048,
         reasoning_effort: str | None = None, retries: int = 6) -> str:
    """Return assistant text for a chat request against an NRP model. Retries
    transient errors with escalating backoff. `reasoning_effort` is accepted for
    interface parity with azure_llm but not forwarded (NRP models ignore it).

    Qwen3 models are reasoning models that otherwise spend the whole token budget
    inside a <think> block and return empty content; we disable thinking so they
    emit the code directly (falls back automatically if the server rejects the
    template kwarg)."""
    extra = {}
    lname = model.lower()
    if "qwen3" in lname:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    last_err = None
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=0.2, **extra)
            _last_meta.clear()
            _last_meta.update(_envelope(resp))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - unified retry/backoff
            last_err = e
            if extra and ("template" in str(e).lower() or "kwarg" in str(e).lower()
                          or "enable_thinking" in str(e)):
                extra = {}          # server rejects the kwarg: retry without it
                continue
            time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f"nrp chat failed after {retries} attempts: {last_err}")


if __name__ == "__main__":
    print("models:", deployments())
    try:
        r = chat("qwen3-small",
                 [{"role": "user", "content": "Reply with exactly: OK"}],
                 max_tokens=50)
        print("chat OK ->", repr(r))
        print("envelope ->", last_meta())
    except Exception as e:  # noqa: BLE001
        print("chat FAILED ->", e)
