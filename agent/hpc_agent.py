"""
hpc_agent.py  --  the grounded agentic diagnostician for LLM-generated parallel code.

This is the HPC counterpart of the churn project's "agentic SHAP": a tool-calling LLM that
answers questions about a generated OpenMP program by CALLING real-measurement tools over the
frozen run pool (diag_core) and narrating only what those tools return. It never asserts a
speedup, a race verdict, or a "correct but not parallel" judgement that a measurement did not
produce -- the same discipline as "never invent a SHAP attribution", ported to parallelism.

Two agents, mirroring the churn orchestrator:
  * the diagnostician  -- reads a program's whole measured story and explains it in plain language
  * the performance-critic -- re-reads the tool facts and FAILs any claim not grounded in them

Auth reuses harness.azure_llm (Entra ID bearer token from the az CLI; no api keys, no
azure-identity). Runs offline against the frozen measurements -- no VM, no compiler at ask time.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import diag_core as DC  # noqa: E402


# ------------------------------------------------------------------- tools ------
class Tools:
    def __init__(self, sess: DC.Session):
        self.s = sess

    def read_program(self, selector) -> dict:
        return self.s.read_program(selector)

    def scaling_distribution(self) -> dict:
        return self.s.scaling_distribution()

    def reward_comparison(self, selector) -> dict:
        return self.s.reward_comparison(selector)

    def list_programs(self) -> dict:
        return {"programs": self.s.list_programs()}


TOOL_SPECS = [
    {"type": "function", "function": {"name": "read_program", "description":
        "Read ONE accepted program's whole MEASURED story: the generated OpenMP source, the task "
        "spec, whether it is correct at every thread count, its real per-thread wall-times, "
        "self-speedup and parallel efficiency at 8 threads, whether it is slower than serial, the "
        "race-check (Archer/TSan) verdict, the synchronization idiom the model chose, the "
        "correctness-only vs efficiency-gated reward, and the static serial-projection result.",
        "parameters": {"type": "object", "properties": {"selector": {"type": "string",
            "description": "task name/substring, model name, a program index, or one of "
                           "'slowest'/'fastest'/'below-serial'/'median'"}}, "required": ["selector"]}}},
    {"type": "function", "function": {"name": "scaling_distribution", "description":
        "Population view across all accepted programs: the self-speedup range the correctness-only "
        "reward cannot see (min/max/median, %% slower than serial, %% below 2x, widest within-task "
        "spread). Use for 'what does correctness miss' / global questions.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "reward_comparison", "description":
        "For one program, what the correctness-only reward pays vs the efficiency-gated reward, and "
        "what each would pay its pragma-stripped (serial) projection -- i.e. the cost of deleting "
        "the parallelism under each reward.",
        "parameters": {"type": "object", "properties": {"selector": {"type": "string"}},
                       "required": ["selector"]}}},
    {"type": "function", "function": {"name": "list_programs", "description":
        "List the accepted+timed programs (index, task, model, self-speedup, verdict) so a specific "
        "one can be chosen.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM_PROMPT = (
    "You are a GROUNDED HPC code diagnostician. You explain LLM-generated OpenMP programs by "
    "OBSERVING real measurements returned by your tools -- correctness at every thread count, "
    "per-thread wall-times, self-speedup and efficiency at 8 threads, race-check verdicts, the "
    "synchronization idiom, and the correctness-only vs efficiency-gated reward. Ground EVERY "
    "claim in a tool result; never invent or estimate a speedup, a race verdict, or a reward "
    "number. If a program is correct at every thread count but its measured self-speedup is near "
    "or below 1, say plainly that it is 'correct but not parallel' and give the measured numbers. "
    "The central point you help people see: an acceptance test that only checks output equivalence "
    "pays a serial program the identical reward as a well-scaled one -- so correctness alone cannot "
    "tell them apart, and the measurements can. Report honest nulls ('scaling not measured') rather "
    "than guessing. Be concise, specific, and quote the numbers you used."
)


def _dispatch(tools: Tools, name, args):
    fn = {
        "read_program": lambda a: tools.read_program(a.get("selector", "slowest")),
        "scaling_distribution": lambda a: tools.scaling_distribution(),
        "reward_comparison": lambda a: tools.reward_comparison(a.get("selector", "slowest")),
        "list_programs": lambda a: tools.list_programs(),
    }[name]
    return fn(args)


# --------------------------------------------------------------- azure client ---
def _client_and_deployment():
    """Reuse the harness Entra-token client; pick a strong chat-route deployment."""
    try:
        from harness import azure_llm
    except Exception as e:  # noqa: BLE001
        print("[warn] azure_llm import failed:", e)
        return None, None
    try:
        client = azure_llm._client()
        deps = azure_llm.deployments()
    except Exception as e:  # noqa: BLE001
        print("[warn] could not build Azure client:", e)
        return None, None
    # prefer a general chat model; avoid responses-only codex/pro routes for tool-calling
    prefer = [d for d in deps if "codex" not in d and "pro" not in d]
    order = ["gpt-5.2", "gpt-5.4", "gpt-5.3", "gpt-4.1", "gpt-4o"]
    for want in order:
        for d in prefer:
            if d.startswith(want):
                return client, d
    return client, (prefer[0] if prefer else (deps[0] if deps else None))


def _chat(client, deployment, system, user, temperature=0.1, json_mode=False):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = client.chat.completions.create(model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    return r.choices[0].message.content or ""


# --------------------------------------------------------------- the critic -----
_CRITIC_SYS = (
    "You STRICTLY verify a grounded HPC diagnosis. You are given the QUESTION, the TOOL FACTS "
    "(JSON -- the only source of truth), and the DRAFT answer. Check: every speedup, efficiency, "
    "wall-time, reward value, race verdict, and 'correct but not parallel' claim in the draft "
    "appears in the facts; the efficiency-gated reward is never described as a calibrated "
    "probability; the serial-projection is described as static (not re-executed). Return JSON "
    '{"verdict":"PASS"|"WARN"|"FAIL","issues":[...],"notes":""}. FAIL only for a NUMBER or a '
    "VERDICT the facts do not support (an invented speedup, a claimed race the check did not find, "
    "or asserting scaling that was not measured).")


def critique(client, deployment, question, facts, answer) -> dict:
    try:
        js = json.loads(_chat(client, deployment, _CRITIC_SYS,
            "QUESTION:\n%s\n\nTOOL FACTS:\n%s\n\nDRAFT:\n%s"
            % (question, json.dumps(facts, default=str)[:12000], answer),
            temperature=0, json_mode=True))
        v = str(js.get("verdict", "WARN")).upper()
        return {"verdict": v if v in ("PASS", "WARN", "FAIL") else "WARN",
                "issues": js.get("issues", [])[:6], "notes": js.get("notes", "")}
    except Exception as e:  # noqa: BLE001
        return {"verdict": "WARN", "issues": ["critic error: %s" % e], "notes": ""}


# --------------------------------------------------------------- the loop -------
def ask_llm(question, sess, client, deployment, max_rounds=6, trace=False) -> dict:
    tools = Tools(sess)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    facts, used = {}, []
    for _ in range(max_rounds):
        resp = client.chat.completions.create(model=deployment, messages=messages,
                                              tools=TOOL_SPECS, tool_choice="auto", temperature=0.1)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "facts": facts, "tool_calls": used}
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                payload = _dispatch(tools, tc.function.name, args)
            except Exception as e:  # noqa: BLE001
                args, payload = {}, {"error": str(e)}
            facts["%s(%s)" % (tc.function.name, args.get("selector", ""))] = payload
            used.append({"tool": tc.function.name, "args": args})
            if trace:
                print("  [tool]", tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(payload, default=str)})
    ans = (client.chat.completions.create(model=deployment, messages=messages, temperature=0.1)
           .choices[0].message.content or "")
    return {"answer": ans, "facts": facts, "tool_calls": used}


def diagnose(question, sess=None, verify=True, trace=False) -> dict:
    """One grounded, critic-verified diagnosis. Returns answer + tool trace + critic verdict."""
    sess = sess or DC.Session(verbose=False)
    client, dep = _client_and_deployment()
    if client is None:
        return {"answer": "(no LLM client -- run `az login`)", "facts": {}, "tool_calls": [],
                "critic": {"verdict": "N/A"}, "deployment": None}
    out = ask_llm(question, sess, client, dep, trace=trace)
    out["deployment"] = dep
    out["critic"] = critique(client, dep, question, out["facts"], out["answer"]) if verify else {"verdict": "SKIPPED"}
    return out


if __name__ == "__main__":
    sess = DC.Session()
    q = " ".join(sys.argv[1:]) or "Take the slowest-scaling accepted program. Is it correct? Is it parallel? What does an output-only acceptance test miss about it?"
    print("\nQ:", q)
    out = diagnose(q, sess, trace=True)
    print("\nmodel:", out["deployment"], "| critic:", out["critic"]["verdict"])
    if out["critic"].get("issues"):
        print("issues:", out["critic"]["issues"])
    print("\nA:", out["answer"])
