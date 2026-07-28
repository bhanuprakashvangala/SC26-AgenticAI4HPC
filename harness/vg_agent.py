"""VG-RLPT: a Verification-Guided agent that writes parallel code under TWO gates.

This is the paper's constructive contribution and the HPC instance of the
Verification-Guided RL Post-Training framework. Generation is a control problem
decomposed into temporally-abstracted Options (Sutton-Precup-Singh), and each Option
terminates only when an *execution verifier* confirms success (execution-based
termination beta):

  Meta-policy pi_meta:  omega_correct  -->  omega_parallel

  Option omega_correct  (the correctness gate, = differential_verify)
      propose -> differentially verify across the thread sweep -> repair on failure.
      beta_correct fires when the program is robustly correct at every thread count.

  Option omega_parallel (the performance gate, this paper's addition)
      given a robust program, propose an optimization -> RE-verify correctness (it must
      stay robust) -> measure self-speedup -> repair on failure.
      beta_parallel fires when the program clears a self-speedup target AND is still
      robustly correct.

The point the paper makes: an agent with ONLY omega_correct (everyone's execute-and-fix
loop) accepts "correct but not parallel" code, because its reward -- correctness -- is
maximized by serial code. Adding omega_parallel makes the reward two-axis and closes
that gap. The final two-axis reward (harness/rewards.py) scores the trajectory.

This module reuses the *same* model client and the *same* verifier as the main study
(run_pareval_batch) plus the performance gate (perf_gate), so a VG-RLPT run is directly
comparable to the correctness-only baseline. It needs model access (Azure/NRP) and the
verification backend (VERIFY_BACKEND=nautilus); run it on the machine that has them.

Usage:
  python -m harness.vg_agent --models gpt-5.4 --tasks 21_histogram_bin_0-100 \
      --correct-budget 2 --perf-budget 2 --target 2.0
Output: results/vg_runs.jsonl  and  logs/pareval/transcripts/vg__*.json
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

from harness import run_pareval_batch as rpb
from harness import perf_gate
from harness import rewards

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "logs" / "pareval" / "transcripts"
OUT = Path(os.environ.get("VG_RESULTS", str(ROOT / "results" / "vg_runs.jsonl")))
THREADS, REPS = rpb.THREADS, rpb.REPS

OPTIMIZE_SYS = (
    "The function below is CORRECT at every thread count but does not scale well. "
    "Rewrite it to run FASTER as threads increase, WITHOUT breaking correctness at any "
    "thread count. Prefer a reduction or a per-thread private accumulator merged once "
    "over a per-element atomic or a critical section inside the hot loop; give each "
    "thread enough independent work; avoid false sharing. Return the COMPLETE function "
    "in one ```cpp block, same signature, no main().")


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _verify_correct(cell) -> dict:
    """Run the correctness gate (full differential sweep) on one cell."""
    res = rpb.batch_verify([cell], THREADS, REPS, f"vg_{cell.id}")
    return res.get(cell.id, {"verdict": "PARSE_ERR"})


def omega_correct(cell, budget: int, trace: list) -> dict:
    """Correctness Option: generate + differential-verify + repair until robust."""
    rpb.gen(cell)
    for rnd in range(budget + 1):
        v = _verify_correct(cell)
        trace.append({"option": "correct", "round": rnd, "verdict": v.get("verdict"),
                      "sweep": v.get("sweep", ""), "diag": v.get("diag", ""),
                      "code": cell.code})
        if rewards.is_robust(v.get("verdict", "")) or rnd == budget:
            return v
        cell.repaired = True
        cell.messages.append({"role": "user", "content":
            f"Your completion failed verification: {v.get('verdict')}. "
            f"{v.get('diag','output depends on thread scheduling (race/order-dependence)')}. "
            "Return the corrected complete function in one ```cpp block."})
        rpb.gen(cell)
    return v


def omega_parallel(cell, task, budget: int, target: float, trace: list) -> perf_gate.PerfResult:
    """Performance Option: optimize for speedup while re-checking correctness.

    Invariant: we only ever accept a candidate that is STILL robustly correct. If an
    optimization breaks correctness or fails to compile, we keep the last good program.
    """
    good_code = cell.code
    res = perf_gate.time_program(good_code, cell.name, task["type"], model=cell.model, target=target)
    trace.append({"option": "parallel", "round": 0, "speedup8": res.speedup8,
                  "scales": res.scales, "valid": res.valid, "code": good_code})
    if res.scales:
        return res

    for rnd in range(1, budget + 1):
        cell.messages.append({"role": "system", "content": OPTIMIZE_SYS})
        cell.messages.append({"role": "user", "content": perf_gate.gate_feedback(res, target)})
        rpb.gen(cell)                                   # produces cell.code = candidate
        cv = _verify_correct(cell)                      # correctness guard
        if not rewards.is_robust(cv.get("verdict", "")):
            trace.append({"option": "parallel", "round": rnd, "rejected": "broke_correctness",
                          "verdict": cv.get("verdict"), "code": cell.code})
            cell.code = good_code                       # revert; try again with feedback
            res_prev = res
            res = perf_gate.PerfResult(True, "PASS", res_prev.speedup8, res_prev.vs_serial8,
                                       res_prev.times, res_prev.scales)
            continue
        res = perf_gate.time_program(cell.code, cell.name, task["type"],
                                     model=cell.model, target=target)
        trace.append({"option": "parallel", "round": rnd, "speedup8": res.speedup8,
                      "scales": res.scales, "valid": res.valid, "code": cell.code})
        if res.robust_timing and res.speedup8 and (not good_code or
                res.speedup8 >= (perf_gate.time_program(good_code, cell.name, task["type"],
                                 model=cell.model, target=target).speedup8 or 0)):
            good_code = cell.code                       # keep the faster correct version
        if res.scales:
            break
    cell.code = good_code
    return perf_gate.time_program(good_code, cell.name, task["type"], model=cell.model, target=target)


def run_task(task_name, task, model, correct_budget, perf_budget, target) -> dict:
    """pi_meta: run the correctness gate, then (if it passed) the performance gate."""
    cell = rpb.Cell(task_name, task, model, "vg", 0)
    trace: list = []

    cv = omega_correct(cell, correct_budget, trace)
    robust = rewards.is_robust(cv.get("verdict", ""))
    sweep = cv.get("sweep", "")

    perf = None
    if robust:
        perf = omega_parallel(cell, task, perf_budget, target, trace)

    speedup = perf.speedup8 if perf else None
    verdict = cv.get("verdict")
    row = {
        "ts": _ts(), "task": task_name, "type": task["type"], "model": model,
        "framework": "vg_rlpt",
        "robust": robust, "verdict": verdict, "sweep": sweep,
        "speedup8": speedup, "scales": bool(perf and perf.scales),
        "reward_correctness_only": rewards.correctness_only(verdict, sweep),
        "reward_two_axis": rewards.two_axis(verdict, speedup, sweep),
        "n_correct_rounds": sum(1 for s in trace if s["option"] == "correct"),
        "n_parallel_rounds": sum(1 for s in trace if s["option"] == "parallel") - 1,
        "final_code": cell.code,
    }
    TR.mkdir(parents=True, exist_ok=True)
    (TR / f"vg__{cell.id}.json").write_text(
        json.dumps({**row, "trace": trace, "messages": cell.messages}, indent=2),
        encoding="utf-8")
    return row


def main():
    tasks = rpb._load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-5.4"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--correct-budget", type=int, default=2)
    ap.add_argument("--perf-budget", type=int, default=2)
    ap.add_argument("--target", type=float, default=2.0,
                    help="8-thread self-speedup a program must reach to clear the perf gate")
    args = ap.parse_args()

    names = args.tasks or [n for n, t in tasks.items()
                           if (not args.types or t["type"] in args.types)]
    print(f"VG-RLPT over {len(names)} tasks x {len(args.models)} models "
          f"(correct budget {args.correct_budget}, perf budget {args.perf_budget}, "
          f"target {args.target}x)")
    with OUT.open("a", encoding="utf-8") as fh:
        for m in args.models:
            for n in names:
                row = run_task(n, tasks[n], m, args.correct_budget, args.perf_budget, args.target)
                fh.write(json.dumps({k: v for k, v in row.items() if k != "final_code"}) + "\n")
                fh.flush()
                s = f"{row['speedup8']:.2f}x" if row["speedup8"] else "n/a"
                print(f"  {n:34s} {m:10s} robust={row['robust']!s:5s} "
                      f"speedup={s:>6s} scales={row['scales']} "
                      f"R2={row['reward_two_axis']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
