"""Function-calling adapter over the keyless Azure client.

harness/azure_llm.chat() returns only assistant text, which is why a scripted
propose->verify->repair loop was the ceiling. Real agents need the model to *choose*
tool calls. This adapter reuses azure_llm's authenticated client, token refresh, and
response-envelope logging, but exposes the full assistant message -- content AND
tool_calls -- so an agent loop can execute the tools the model asked for and feed the
results back.

Targets the chat-completions tool API. Deployments served only through the Responses
API (pro/codex) would need the responses-tools variant; the agents here use the
tool-capable chat deployments (gpt-5.x). Open-weights NRP models can be added once their
gateway advertises tool support.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field

from harness import azure_llm


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Assistant:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def complete(deployment: str, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto", max_tokens: int = 4096,
             reasoning_effort: str | None = None, retries: int = 6) -> Assistant:
    """One assistant turn. If `tools` is given, the model may return tool_calls.

    Mirrors azure_llm.chat()'s retry/backoff and 401 token-refresh so long agent runs
    survive token rotation and transient endpoint errors.
    """
    last_err = None
    for attempt in range(retries):
        try:
            client = azure_llm._client()
            kw: dict = {"model": deployment, "messages": messages,
                        "max_completion_tokens": max_tokens}
            if tools:
                kw["tools"] = tools
                kw["tool_choice"] = tool_choice
            if reasoning_effort:
                kw["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**kw)
            azure_llm._last_meta.clear()
            azure_llm._last_meta.update(azure_llm._envelope(resp))
            m = resp.choices[0].message
            calls = []
            for tc in (getattr(m, "tool_calls", None) or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return Assistant(content=(m.content or ""), tool_calls=calls,
                             meta=azure_llm.last_meta())
        except Exception as e:  # noqa: BLE001 - unified retry
            last_err = e
            msg = str(e)
            if "401" in msg or "expired" in msg.lower() or "Unauthorized" in msg:
                azure_llm._refresh_client()
                time.sleep(2)
                continue
            time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f"tool-completion failed after {retries} attempts: {last_err}")


def assistant_message(a: Assistant) -> dict:
    """Serialize an Assistant back into an OpenAI chat message (for the next turn)."""
    msg: dict = {"role": "assistant", "content": a.content or None}
    if a.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in a.tool_calls
        ]
    return msg


def tool_message(call: ToolCall, result: str) -> dict:
    """The tool-result message the model reads on the next turn."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result}
