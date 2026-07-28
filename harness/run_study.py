"""Resumable study driver: run the agentic ParEval ablation ONE TASK AT A TIME.

Why per-task invocations? run_pareval_batch writes result rows only at the very
end of an invocation, so a single monolithic run that hangs (a cold VM
run-command, a laptop sleep) loses everything. Driving it one task per subprocess
means each finished task's 12 cells (4 models x 3 conditions) are flushed to
results/pareval_runs.jsonl before the next task starts, and the harness's own
skip-resume (it reads already-done (task,model,condition,trial) tuples) makes the
whole study restartable. A per-task subprocess timeout turns a hung VM call into a
skipped task instead of a wedged study.

Usage:
  python -m harness.run_study                 # default expanded task set, 4 models
  python -m harness.run_study --trials 2      # add a second generation trial
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "enrich_driver.log"

MODELS = ["gpt-5.4", "gpt-5.2", "gpt-5.3-codex", "gpt-5.4-pro"]

# Existing 8 tasks (already in results) + 12 new spanning all 12 ParEval OMP
# problem types, weighted to race-prone families. The harness skips done cells,
# so listing the existing tasks here is a no-op that documents the full study.
EXISTING = [
    "17_graph_highest_degree", "21_histogram_bin_0-100",
    "22_histogram_count_quadrants", "26_reduce_product_of_inverses",
    "29_reduce_sum_of_min_of_pairs", "33_scan_reverse_prefix_sum",
    "35_search_search_for_last_struct_by_key", "38_search_find_the_first_even_number",
]
NEW = [
    "20_histogram_pixel_histogram", "25_reduce_xor", "30_scan_prefix_sum",
    "15_graph_edge_count", "36_search_check_if_array_contains_value",
    "40_sort_sort_an_array_of_complex_numbers_by_magnitude",
    "45_sparse_la_sparse_solve", "00_dense_la_lu_decomp", "05_fft_inverse_fft",
    "10_geometry_convex_hull", "50_stencil_xor_kernel", "55_transform_relu",
]
ALL_TASKS = NEW + EXISTING   # race-prone new tasks first (most informative)


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--per-task-timeout", type=int, default=1500)
    args = ap.parse_args()

    log(f"STUDY START: {len(args.tasks)} tasks x {len(args.models)} models "
        f"x {args.trials} trial(s); timeout {args.per_task_timeout}s/task")
    for i, task in enumerate(args.tasks, 1):
        t0 = time.time()
        cmd = [sys.executable, "-m", "harness.run_pareval_batch",
               "--models", *args.models, "--tasks", task,
               "--trials", str(args.trials)]
        log(f"[{i}/{len(args.tasks)}] {task} -> {' '.join(cmd[2:])}")
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                               timeout=args.per_task_timeout)
            tail = "\n".join((r.stdout or "").splitlines()[-3:])
            log(f"    done in {time.time()-t0:.0f}s rc={r.returncode}: {tail}")
            if r.returncode != 0:
                log(f"    STDERR tail: {(r.stderr or '').strip()[-300:]}")
        except subprocess.TimeoutExpired:
            log(f"    TIMEOUT after {args.per_task_timeout}s -- skipping (resumable)")
    log("STUDY DONE")


if __name__ == "__main__":
    main()
