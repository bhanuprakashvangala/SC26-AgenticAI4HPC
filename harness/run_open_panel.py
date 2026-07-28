"""Run the open-model (NRP) panel one model at a time, appending to a shared
resumable results file. Per-model invocation means each model's rows persist as
soon as it finishes -- robust to interruption (unlike a single all-models run
that only writes at the very end).
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen3-small", "gpt-oss", "minimax-m2", "gemma"]
TASKS = ["21_histogram_bin_0-100", "22_histogram_count_quadrants",
         "26_reduce_product_of_inverses", "29_reduce_sum_of_min_of_pairs",
         "33_scan_reverse_prefix_sum", "17_graph_highest_degree"]

def main():
    env = dict(os.environ)
    env["VERIFY_BACKEND"] = "nautilus"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("RESULTS_FILE", str(ROOT / "results" / "pareval_open.jsonl"))
    for m in MODELS:
        print(f"\n{'#'*70}\n# MODEL {m}\n{'#'*70}", flush=True)
        cmd = [sys.executable, "-m", "harness.run_pareval_batch",
               "--models", m, "--tasks", *TASKS,
               "--budget", "2", "--trials", "1"]
        r = subprocess.run(cmd, env=env, cwd=str(ROOT))
        print(f"# {m} exit={r.returncode}", flush=True)

if __name__ == "__main__":
    main()
