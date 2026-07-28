"""ToolAgent: a genuine function-calling agent loop.

The model is given tools and a goal. On each turn it either calls tools (which we
execute against the live verifier environment and feed back) or returns a final answer.
This is the ReAct/tool-use pattern -- the agent decides *when* to verify and *when* it
is done -- not a fixed propose->verify->repair script.

Two role agents specialize the loop by system prompt and toolset:
  CoderAgent      (omega_correct)  -- write a correct OpenMP function; must call
                                      compile_and_verify and see it pass before finishing.
  OptimizerAgent  (omega_parallel) -- make a correct function scale; must keep it correct
                                      (compile_and_verify) AND improve measure_speedup.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from harness.agentic import llm
from harness.agentic.tools import ToolBox

_CODE_RE = re.compile(r"```(?:cpp|c\+\+|c)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RE.findall(text or "")
    return (m[-1].strip() if m else "")


@dataclass
class AgentResult:
    final_code: str
    messages: list[dict]
    tool_events: list[dict] = field(default_factory=list)
    steps: int = 0
    stop_reason: str = ""


class ToolAgent:
    def __init__(self, deployment: str, system: str, box: ToolBox,
                 tool_names: list[str] | None = None, max_steps: int = 8,
                 max_tokens: int = 4096):
        self.deployment = deployment
        self.system = system
        self.box = box
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        all_tools = box.openai_tools()
        self.tools = ([t for t in all_tools if t["function"]["name"] in tool_names]
                      if tool_names else all_tools)

    def run(self, user: str, seed_messages: list[dict] | None = None) -> AgentResult:
        messages = seed_messages or [{"role": "system", "content": self.system}]
        messages.append({"role": "user", "content": user})
        events: list[dict] = []
        last_code = ""

        for step in range(self.max_steps):
            a = llm.complete(self.deployment, messages, tools=self.tools,
                             max_tokens=self.max_tokens)
            messages.append(llm.assistant_message(a))

            if a.content and extract_code(a.content):
                last_code = extract_code(a.content)

            if not a.wants_tools:
                # model produced a final answer; keep the best code we have seen
                code = extract_code(a.content) or last_code or self.box.last_robust_code or ""
                return AgentResult(code, messages, events, step + 1, "final_answer")

            for call in a.tool_calls:
                result = self.box.dispatch(call.name, call.arguments)
                events.append({"step": step, "tool": call.name,
                               "args": {k: v for k, v in call.arguments.items() if k != "code"},
                               "result": result})
                messages.append(llm.tool_message(call, result))
                # if the model just verified some code, remember it
                if "code" in call.arguments:
                    last_code = call.arguments["code"] or last_code

        # budget exhausted: return the last robust code if any, else last seen
        code = self.box.last_robust_code or last_code
        return AgentResult(code, messages, events, self.max_steps, "budget_exhausted")


# ------------------------------ role definitions ------------------------------
CODER_SYS = (
    "You are an expert HPC engineer writing OpenMP C++. Implement the requested function so "
    "it is correct under EVERY thread count. You have tools: call `task_spec` if you need the "
    "exact signature, and you MUST call `compile_and_verify` and see verdict ROBUST_CORRECT "
    "before you finish. If it reports a failing thread count, that is a data race or "
    "order-dependence -- fix it (reduction, atomic, private accumulator, or a total order) and "
    "verify again. Never write a shared scalar or array element from multiple threads without "
    "synchronization. When verification passes, reply with the final function in one ```cpp "
    "block and no tool call.")

OPTIMIZER_SYS = (
    "You are an expert HPC performance engineer. You are given a CORRECT OpenMP function that "
    "does not scale well. Make it run FASTER as threads increase WITHOUT breaking correctness. "
    "Use `measure_speedup` to see its 8-thread self-speedup and `compile_and_verify` to confirm "
    "it is still correct at every thread count after each change. Prefer a reduction or a "
    "per-thread private accumulator merged once over a per-element atomic or a critical section "
    "in the hot loop; give each thread enough independent work; avoid false sharing. Iterate "
    "until it clears the speedup target while remaining ROBUST_CORRECT, then reply with the "
    "final function in one ```cpp block and no tool call.")


def coder_agent(deployment: str, box: ToolBox, max_steps: int = 8) -> ToolAgent:
    return ToolAgent(deployment, CODER_SYS, box,
                     tool_names=["task_spec", "compile_and_verify"], max_steps=max_steps)


def optimizer_agent(deployment: str, box: ToolBox, max_steps: int = 8) -> ToolAgent:
    return ToolAgent(deployment, OPTIMIZER_SYS, box,
                     tool_names=["compile_and_verify", "measure_speedup"], max_steps=max_steps)
