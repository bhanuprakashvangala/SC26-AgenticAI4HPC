"""VG-RLPT agentic stack: tool-calling agents orchestrated as a state graph.

Layers:
  llm.py     - function-calling adapter over the keyless Azure client (returns tool_calls)
  tools.py   - the verifier ENVIRONMENT exposed as callable tools the model invokes itself
  agent.py   - ToolAgent: a real ReAct/function-calling loop (the model drives tool use)
  graph.py   - the orchestrator: a LangGraph state machine implementing the meta-policy
               omega_correct -> omega_parallel with gate-guarded routing
  run.py     - CLI entry point over ParEval tasks

This is the "real pipeline": agents choose actions (tool calls) autonomously against a
live verification environment, and a stateful graph routes between the two gates.
"""
