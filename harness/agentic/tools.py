"""The verification ENVIRONMENT, exposed as tools the agent invokes itself.

Instead of a hardcoded verify step, the agent is handed real tools and decides when to
call them. Each tool executes against the live backend the study uses -- the
differential correctness verifier (run_pareval_batch.batch_verify) and the performance
gate (perf_gate) -- and returns a compact observation the model reads on its next turn.

A ToolBox is bound to one (task, model): it carries the task context and remembers the
last program that passed the correctness gate, so the orchestrator can always fall back
to a known-good version. `openai_tools()` returns the function-calling schemas;
`dispatch()` executes a call by name.
"""
from __future__ import annotations
import json

from harness import run_pareval_batch as rpb
from harness import perf_gate
from harness import rewards


class ToolBox:
    def __init__(self, task_name: str, task: dict, model: str):
        self.task_name = task_name
        self.task = task
        self.model = model
        self.calls: list[dict] = []           # audit log of every tool invocation
        self.last_robust_code: str | None = None
        self.last_verdict: str | None = None
        self.last_sweep: str = ""
        self.last_speedup: float | None = None

    # ---------------------------- tool implementations ----------------------------
    def compile_and_verify(self, code: str, threads: list[int] | None = None,
                           reps: int = 2) -> str:
        """Compile the code and differentially verify it across a thread sweep against
        the trusted serial reference. threads defaults to {1,2,4,8}."""
        threads = threads or rpb.THREADS
        cell = rpb.Cell(self.task_name, self.task, self.model, "agentic", len(self.calls))
        cell.code = code
        res = rpb.batch_verify([cell], threads, reps, f"ag_{cell.id}")
        r = res.get(cell.id, {"verdict": "PARSE_ERR"})
        verdict, sweep, diag = r.get("verdict"), r.get("sweep", ""), r.get("diag", "")
        self.last_verdict, self.last_sweep = verdict, sweep
        if rewards.is_robust(verdict):
            self.last_robust_code = code
        self.calls.append({"tool": "compile_and_verify", "verdict": verdict, "sweep": sweep})
        out = {"verdict": verdict, "per_thread_pass": sweep or "n/a", "diagnostic": diag or "none",
               "robust": rewards.is_robust(verdict)}
        return json.dumps(out)

    def measure_speedup(self, code: str, target: float = 2.0) -> str:
        """Measure the program's 8-thread self-speedup (only meaningful once correct)."""
        res = perf_gate.time_program(code, self.task_name, self.task["type"],
                                     model=self.model, target=target)
        self.last_speedup = res.speedup8
        self.calls.append({"tool": "measure_speedup", "speedup8": res.speedup8,
                           "scales": res.scales})
        out = {"compiled": res.compiled, "valid": res.valid, "self_speedup_8": res.speedup8,
               "vs_serial_8": res.vs_serial8, "scales_target": target,
               "clears_gate": res.scales, "hint": perf_gate.gate_feedback(res, target)}
        return json.dumps(out)

    def task_spec(self) -> str:
        """Return the task prompt/signature the function must implement."""
        return json.dumps({"task": self.task_name, "family": self.task["type"],
                           "prompt": self.task["prompt"]})

    # ------------------------------- tool registry -------------------------------
    def openai_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": "compile_and_verify",
                "description": ("Compile the OpenMP function and check it against the trusted "
                                "serial reference at each thread count with repeats. Use this "
                                "to confirm correctness before finishing; a race shows up as a "
                                "thread count that fails."),
                "parameters": {"type": "object", "properties": {
                    "code": {"type": "string", "description": "the complete C++ function"},
                    "threads": {"type": "array", "items": {"type": "integer"},
                                "description": "thread counts to test; default [1,2,4,8]"},
                    "reps": {"type": "integer", "description": "repeats per thread count (default 2)"},
                }, "required": ["code"]}}},
            {"type": "function", "function": {
                "name": "measure_speedup",
                "description": ("Time the function across threads and report 8-thread "
                                "self-speedup. Only meaningful for a correct program. Use this "
                                "to check whether the code actually scales."),
                "parameters": {"type": "object", "properties": {
                    "code": {"type": "string", "description": "the complete C++ function"},
                    "target": {"type": "number", "description": "speedup target (default 2.0)"},
                }, "required": ["code"]}}},
            {"type": "function", "function": {
                "name": "task_spec",
                "description": "Return the exact task prompt and function signature to implement.",
                "parameters": {"type": "object", "properties": {}}}},
        ]

    def dispatch(self, name: str, args: dict) -> str:
        if name == "compile_and_verify":
            return self.compile_and_verify(args.get("code", ""), args.get("threads"),
                                           int(args.get("reps", 2)))
        if name == "measure_speedup":
            return self.measure_speedup(args.get("code", ""), float(args.get("target", 2.0)))
        if name == "task_spec":
            return self.task_spec()
        return json.dumps({"error": f"unknown tool {name}"})
