r"""The orchestrator: the meta-policy as a stateful graph.

This wires the two tool-calling agents into VG-RLPT's control flow:

        START
          |
     [coder_node]         omega_correct: CoderAgent writes + verifies until robust
          |
     route_after_coder    beta_correct? robust -> optimize ; not robust -> judge (fail)
        /       \
 [optimizer_node] |       omega_parallel: OptimizerAgent makes it scale, staying correct
        \       /
      [judge_node]        compute the two-axis reward from the final (verdict, speedup)
          |
         END

Implemented on LangGraph's StateGraph when installed (the real stack); falls back to an
equivalent hand-rolled sequential driver otherwise, so it runs either way. State is a
plain dict so both paths share the node functions.
"""
from __future__ import annotations
from typing import Any

from harness.agentic.tools import ToolBox
from harness.agentic import agent as agents
from harness import rewards


# ------------------------------- node functions -------------------------------
def coder_node(state: dict) -> dict:
    """omega_correct: run the coder agent, then authoritatively verify its output."""
    box: ToolBox = state["box"]
    ag = agents.coder_agent(state["model"], box, max_steps=state.get("correct_steps", 8))
    res = ag.run(f"Implement this task. Call task_spec if needed.\n\n{box.task['prompt']}")
    # authoritative correctness check of the agent's final program
    if res.final_code:
        box.compile_and_verify(res.final_code)
    verdict = box.last_verdict or "PARSE_ERR"
    return {
        "code": box.last_robust_code or res.final_code,
        "verdict": verdict,
        "sweep": box.last_sweep,
        "correct": rewards.is_robust(verdict),
        "trace": state.get("trace", []) + [{"phase": "omega_correct",
            "stop": res.stop_reason, "steps": res.steps, "tools": res.tool_events}],
        "coder_messages": res.messages,
    }


def route_after_coder(state: dict) -> str:
    return "optimize" if state.get("correct") else "finish"


def optimizer_node(state: dict) -> dict:
    """omega_parallel: run the optimizer agent on the robust program; keep the fastest
    version that is STILL correct (the performance gate is guarded by correctness)."""
    box: ToolBox = state["box"]
    good = box.last_robust_code or state["code"]
    ag = agents.optimizer_agent(state["model"], box, max_steps=state.get("perf_steps", 8))
    res = ag.run("This function is correct but does not scale well. Improve its 8-thread "
                 "self-speedup while keeping it correct at every thread count. Current "
                 f"function:\n```cpp\n{good}\n```")
    cand = res.final_code or good
    # correctness guard on the optimized candidate
    box.compile_and_verify(cand)
    final = cand if rewards.is_robust(box.last_verdict or "") else good
    perf = box.measure_speedup(final, target=state.get("target", 2.0))
    import json
    pj = json.loads(perf)
    return {
        "code": final,
        "verdict": "ROBUST_CORRECT",     # guaranteed: we reverted if the candidate broke
        "speedup8": pj.get("self_speedup_8"),
        "scales": bool(pj.get("clears_gate")),
        "trace": state.get("trace", []) + [{"phase": "omega_parallel",
            "stop": res.stop_reason, "steps": res.steps, "tools": res.tool_events,
            "reverted": final is good}],
        "optimizer_messages": res.messages,
    }


def judge_node(state: dict) -> dict:
    """Score the trajectory with both rewards -- the whole point of the pipeline."""
    verdict = state.get("verdict", "PARSE_ERR")
    sweep = state.get("sweep", "")
    speedup = state.get("speedup8")
    return {
        "reward_correctness_only": rewards.correctness_only(verdict, sweep),
        "reward_two_axis": rewards.two_axis(verdict, speedup, sweep),
        "done": True,
    }


# ------------------------------- graph assembly -------------------------------
def _build_langgraph():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(dict)
    g.add_node("coder", coder_node)
    g.add_node("optimizer", optimizer_node)
    g.add_node("judge", judge_node)
    g.add_edge(START, "coder")
    g.add_conditional_edges("coder", route_after_coder,
                            {"optimize": "optimizer", "finish": "judge"})
    g.add_edge("optimizer", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_pipeline(task_name: str, task: dict, model: str, *, correct_steps: int = 8,
                 perf_steps: int = 8, target: float = 2.0) -> dict:
    """Run the full VG-RLPT pipeline for one (task, model). Uses LangGraph if present,
    otherwise an equivalent sequential driver."""
    state: dict[str, Any] = {
        "task_name": task_name, "task": task, "model": model,
        "box": ToolBox(task_name, task, model),
        "correct_steps": correct_steps, "perf_steps": perf_steps, "target": target,
        "trace": [],
    }
    try:
        app = _build_langgraph()
        final = app.invoke(state)
    except ImportError:
        # hand-rolled fallback: same nodes, same routing
        final = dict(state)
        final.update(coder_node(final))
        if route_after_coder(final) == "optimize":
            final.update(optimizer_node(final))
        final.update(judge_node(final))
    # strip non-serializable bits before returning a clean result row
    final.pop("box", None)
    return final
