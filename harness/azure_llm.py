"""Azure OpenAI access via AAD tokens obtained from the az CLI.

The corporate resource has local-auth (API keys) disabled, so we authenticate
with Entra ID bearer tokens. To avoid the azure-identity dependency (which needs
a Rust build of `cryptography` unavailable on this ARM64 host), we shell out to
`az account get-access-token` and cache the token until it nears expiry.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from openai import AzureOpenAI

_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = _ROOT / ".secrets" / "azure_openai.json"
_SCOPE = "https://cognitiveservices.azure.com/.default"
_RESOURCE = "https://cognitiveservices.azure.com"

_cfg = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
_AZ = _cfg.get("az_path", "az")

_token_cache = {"token": None, "exp": 0.0}
_client_cache: dict[str, AzureOpenAI] = {}


def _get_token() -> str:
    now = time.time()
def _get_token(force: bool = False) -> str:
    now = time.time()
    if not force and _token_cache["token"] and _token_cache["exp"] - now > 300:
        return _token_cache["token"]
    out = subprocess.run(
        [_AZ, "account", "get-access-token", "--resource", _RESOURCE,
         "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, shell=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"az token fetch failed: {out.stderr.strip()}")
    tok = out.stdout.strip()
    _token_cache["token"] = tok
    _token_cache["exp"] = now + 2400  # refresh at 40 min (well inside token lifetime)
    return tok


def _refresh_client() -> None:
    """Force a brand-new token + client (used after a 401 during a long run)."""
    _token_cache["token"] = None
    _token_cache["exp"] = 0.0
    _client_cache.clear()


def _client() -> AzureOpenAI:
    tok = _get_token()
    # Rebuild client when the token rotates so the Authorization header is fresh.
    if _client_cache.get("token") != tok:
        _client_cache.clear()
        _client_cache["token"] = tok
        _client_cache["client"] = AzureOpenAI(
            azure_endpoint=_cfg["endpoint"],
            azure_ad_token=tok,
            api_version=_cfg["api_version"],
            timeout=240.0,
            max_retries=0,
        )
    return _client_cache["client"]


def deployments() -> list[str]:
    return list(_cfg["deployments"])


# Some frontier deployments (codex/pro) are served only through the Responses
# API; others use classic chat completions. We seed the known routing and fall
# back automatically if a deployment rejects the chat route.
_api_mode: dict[str, str] = {
    "gpt-5.3-codex": "responses",
    "gpt-5.4-pro": "responses",
}


def _extract_responses_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()
    # Fallback: walk the output items for any text content.
    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts).strip()


def _envelope(resp) -> dict:
    """Extract the metadata the LLM service itself emits — the fields that prove a
    real API call (request id, served model version, token usage, finish reason)."""
    try:
        d = resp.model_dump()
    except Exception:
        return {}
    ch = (d.get("choices") or [{}])[0]
    return {
        "id": d.get("id"),
        "served_model": d.get("model"),
        "created": d.get("created"),
        "system_fingerprint": d.get("system_fingerprint"),
        "usage": d.get("usage"),
        "finish_reason": ch.get("finish_reason") or d.get("status"),
    }


# The metadata the most recent call's LLM response carried. run_pareval_batch reads
# this right after chat() so every generation log records the LLM-emitted envelope.
_last_meta: dict = {}


def last_meta() -> dict:
    return dict(_last_meta)


def _chat_completions(deployment, messages, max_tokens, reasoning_effort):
    kw = {"model": deployment, "messages": messages,
          "max_completion_tokens": max_tokens}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    resp = _client().chat.completions.create(**kw)
    _last_meta.clear(); _last_meta.update(_envelope(resp))
    return (resp.choices[0].message.content or "").strip()


def _responses(deployment, messages, max_tokens, reasoning_effort):
    kw = {"model": deployment, "input": messages,
          "max_output_tokens": max_tokens}
    if reasoning_effort:
        kw["reasoning"] = {"effort": reasoning_effort}
    resp = _client().responses.create(**kw)
    _last_meta.clear(); _last_meta.update(_envelope(resp))
    return _extract_responses_text(resp)


def chat(deployment: str, messages: list[dict], max_tokens: int = 2048,
         reasoning_effort: str | None = None, retries: int = 8) -> str:
    """Return assistant text for a chat request. Routes between chat-completions
    and the Responses API per deployment, and retries transient errors with
    escalating backoff (tolerant of transient network/endpoint degradation)."""
    last_err = None
    for attempt in range(retries):
        mode = _api_mode.get(deployment, "chat")
        try:
            if mode == "responses":
                return _responses(deployment, messages, max_tokens, reasoning_effort)
            return _chat_completions(deployment, messages, max_tokens, reasoning_effort)
        except Exception as e:  # noqa: BLE001 - unified retry/backoff
            last_err = e
            msg = str(e)
            if mode == "chat" and "unsupported" in msg.lower():
                _api_mode[deployment] = "responses"  # switch route, retry now
                continue
            if "401" in msg or "expired" in msg.lower() or "Unauthorized" in msg:
                _refresh_client()   # token expired mid-run: force a fresh token
                time.sleep(2)
                continue
            time.sleep(min(15 * (attempt + 1), 90))  # 15,30,45,...,90s
    raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")


if __name__ == "__main__":
    for dep in deployments():
        try:
            r = chat(dep, [{"role": "user",
                            "content": "Reply with exactly: OK"}], max_tokens=2000)
            print(f"[OK]   {dep}: {r!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {dep}: {e}")
