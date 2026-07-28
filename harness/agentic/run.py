"""Run the VG-RLPT agentic pipeline over ParEval tasks.

Each (task, model) goes through the orchestration graph: the coder agent (omega_correct)
tool-calls the verifier until robust, then -- if it passed -- the optimizer agent
(omega_parallel) tool-calls the timer until it scales, then the judge scores both
rewards. Every tool call and both agents' full message logs are written per run.

Compare against the correctness-only baseline (run_pareval_batch): same models, same
verifier, but no performance gate -- the pipeline here is what adds the second axis.

Usage:
  python -m harness.agentic.run --models gpt-5.4 --types histogram reduce search \
      --correct-steps 8 --perf-steps 8 --target 2.0
Output: results/agentic_runs.jsonl  and  logs/pareval/transcripts/agentic__*.json
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

from harness import run_pareval_batch as rpb
from harness.agentic import graph

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "logs" / "pareval" / "transcripts"
OUT = Path(os.environ.get("AGENTIC_RESULTS", str(ROOT / "results" / "agentic_runs.jsonl")))


def _ts():
    return datetime.now(timezone.utc).isoformat()


def main():
    tasks = rpb._load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-5.4"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--correct-steps", type=int, default=8)
    ap.add_argument("--perf-steps", type=int, default=8)
    ap.add_argument("--target", type=float, default=2.0)
    args = ap.parse_args()

    names = args.tasks or [n for n, t in tasks.items()
                           if (not args.types or t["type"] in args.types)]
    TR.mkdir(parents=True, exist_ok=True)
    print(f"agentic VG-RLPT pipeline: {len(names)} tasks x {len(args.models)} models")

    with OUT.open("a", encoding="utf-8") as fh:
        for m in args.models:
            for n in names:
                st = graph.run_pipeline(n, tasks[n], m,
                                        correct_steps=args.correct_steps,
                                        perf_steps=args.perf_steps, target=args.target)
                row = {"ts": _ts(), "task": n, "type": tasks[n]["type"], "model": m,
                       "framework": "vg_rlpt_agentic",
                       "correct": st.get("correct"), "verdict": st.get("verdict"),
                       "sweep": st.get("sweep"), "speedup8": st.get("speedup8"),
                       "scales": st.get("scales"),
                       "reward_correctness_only": st.get("reward_correctness_only"),
                       "reward_two_axis": st.get("reward_two_axis")}
                # full multi-agent transcript
                cid = f"{n}__{m}__agentic".replace(".", "-")
                (TR / f"agentic__{cid}.json").write_text(json.dumps({
                    **row, "trace": st.get("trace", []),
                    "coder_messages": st.get("coder_messages", []),
                    "optimizer_messages": st.get("optimizer_messages", []),
                    "final_code": st.get("code", "")}, indent=2), encoding="utf-8")
                fh.write(json.dumps(row) + "\n"); fh.flush()
                sp = f"{row['speedup8']:.2f}x" if row["speedup8"] else "n/a"
                print(f"  {n:34s} {m:10s} correct={row['correct']!s:5s} "
                      f"speedup={sp:>6s} scales={row['scales']} "
                      f"R2={row['reward_two_axis']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
