"""Best-of-N reward selection: a controlled proxy for what GRPO/RLVR would learn.

GRPO scores a GROUP of sampled completions with a verifiable reward and pushes the
policy toward the higher-scoring ones. Full RL training is expensive; best-of-N is its
cheap, faithful proxy: sample N completions, score each with a reward, and KEEP the
best. Whatever reward you select by is the reward the policy would chase.

We sample N completions per (task, model), verify each with the SAME differential
correctness gate as the study, time the correct ones with the performance gate, and
then ask a single question: which program does each reward select?

  correctness-only reward -> selects a RANDOM robust program (all robust ones tie), so
                             the realized speedup is, in expectation, the pool mean.
  two-axis reward         -> selects the FASTEST robust program (the pool max).

The gap between them is the speedup a correctness-only RL objective leaves on the
table -- reward hacking, measured directly. This mirrors harness/reward_selection.py
(which runs on the frozen study data) but with LIVE sampling, so it also captures
within-model variance, not just cross-condition variance.

Needs model access (Azure/NRP) and VERIFY_BACKEND=nautilus. Run on the machine that
has them.

Usage:
  python -m harness.best_of_n --models gpt-5.4 --n 8 --types histogram reduce search
Output: results/best_of_n.jsonl
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from harness import run_pareval_batch as rpb
from harness import perf_gate
from harness import rewards

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("BON_RESULTS", str(ROOT / "results" / "best_of_n.jsonl")))
THREADS, REPS = rpb.THREADS, rpb.REPS


def _ts():
    return datetime.now(timezone.utc).isoformat()


def sample_pool(task_name, task, model, n) -> list[dict]:
    """Draw N single-shot completions, verify correctness (batched), time the correct
    ones, and return a scored candidate pool."""
    cells = [rpb.Cell(task_name, task, model, "bon", i) for i in range(n)]
    for c in cells:
        rpb.gen(c)                                  # one sample each
    verd = rpb.batch_verify(cells, THREADS, REPS, f"bon_{task_name}_{model}")

    pool = []
    for c in cells:
        v = verd.get(c.id, {"verdict": "PARSE_ERR"})
        verdict, sweep = v.get("verdict"), v.get("sweep", "")
        speedup = None
        if rewards.is_robust(verdict):
            pr = perf_gate.time_program(c.code, task_name, task["type"], model=model, trial=c.trial)
            speedup = pr.speedup8
        pool.append({
            "trial": c.trial, "verdict": verdict, "sweep": sweep, "speedup8": speedup,
            "robust": rewards.is_robust(verdict),
            "r_correct": rewards.correctness_only(verdict, sweep),
            "r_two_axis": rewards.two_axis(verdict, speedup, sweep),
            "code": c.code,
        })
    return pool


def select(pool) -> dict:
    """Realized speedup under each reward's selection over the candidate pool."""
    robust = [c for c in pool if c["robust"] and c["speedup8"]]
    if not robust:
        return {"n_robust": len(robust), "corr_only_expected": None,
                "corr_only_worst": None, "two_axis": None}
    sp = [c["speedup8"] for c in robust]
    return {
        "n_robust": len(robust),
        "corr_only_expected": mean(sp),          # random robust pick
        "corr_only_worst": min(sp),              # adversarial robust pick
        "two_axis": max(sp),                     # speedup-aware pick
    }


def main():
    tasks = rpb._load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-5.4"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--types", nargs="+", default=None,
                    help="restrict to race-prone families, e.g. histogram reduce search")
    ap.add_argument("--n", type=int, default=8, help="samples per (task, model)")
    args = ap.parse_args()

    names = args.tasks or [n for n, t in tasks.items()
                           if (not args.types or t["type"] in args.types)]
    print(f"best-of-{args.n} over {len(names)} tasks x {len(args.models)} models")

    agg = []
    with OUT.open("a", encoding="utf-8") as fh:
        for m in args.models:
            for n in names:
                pool = sample_pool(n, tasks[n], m, args.n)
                sel = select(pool)
                row = {"ts": _ts(), "task": n, "type": tasks[n]["type"], "model": m,
                       "n": args.n, **sel,
                       "pool": [{k: c[k] for k in ("trial", "verdict", "speedup8", "robust")}
                                for c in pool]}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if sel["two_axis"]:
                    agg.append(sel)
                    print(f"  {n:34s} {m:10s} robust={sel['n_robust']}/{args.n} "
                          f"corr-only~{sel['corr_only_expected']:.2f}x "
                          f"two-axis={sel['two_axis']:.2f}x")
                else:
                    print(f"  {n:34s} {m:10s} no robust+timed candidate")

    if agg:
        co = mean(a["corr_only_expected"] for a in agg)
        ta = mean(a["two_axis"] for a in agg)
        print(f"\nAcross {len(agg)} (task,model) pools with a robust candidate:")
        print(f"  correctness-only reward realizes ~{co:.2f}x (random robust pick)")
        print(f"  two-axis reward realizes          {ta:.2f}x  (+{100*(ta-co)/co:.0f}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
