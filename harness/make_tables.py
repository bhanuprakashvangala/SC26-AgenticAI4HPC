"""Emit in-depth results tables (LaTeX table bodies) from the real logged data:
  paper/table_frontier.tex  -- per-model x tool-level robust correctness + Wilson CI
  paper/table_open.tex       -- open-weights panel: robust rate, races shipped/caught
All numbers are computed from results/*.jsonl so the tables stay in lock-step with
the data (nothing hand-typed).
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

from harness.figstyle import wilson

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "results" / "pareval_runs.jsonl"
OPEN = ROOT / "results" / "pareval_open.jsonl"
COND = ["single_shot", "execute_once", "differential_verify"]
RACE_TASKS = ["21_histogram_bin_0-100", "22_histogram_count_quadrants",
              "26_reduce_product_of_inverses", "29_reduce_sum_of_min_of_pairs",
              "33_scan_reverse_prefix_sum", "17_graph_highest_degree"]


def _load(path):
    seen = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                seen[(r["task"], r["model"], r["condition"], r.get("trial", 0))] = r
    return seen


def _is_race(sweep):
    m = {int(a): (int(b), int(c))
         for a, b, c in re.findall(r"t(\d+)=(\d+)/(\d+)", sweep or "")}
    if not m:
        return False
    p1 = m.get(1, (0, 0))
    return p1[1] > 0 and p1[0] == p1[1] and any(m[t][0] < m[t][1] for t in m if t > 1)


def _pretty(m):
    mapping = {"qwen3-small": "Qwen3-27B", "gpt-oss": "gpt-oss-120B",
               "minimax-m2": "MiniMax-M2", "gemma": "Gemma-31B"}
    return mapping.get(m, m.replace("gpt-", "GPT-"))


def frontier_table(seen):
    models = sorted({k[1] for k in seen})
    rows = []
    for m in models:
        cells = []
        for c in COND:
            sub = [r for r in seen.values() if r["model"] == m and r["condition"] == c]
            k = sum(1 for r in sub if r.get("robust"))
            n = len(sub)
            p, lo, hi = wilson(k, n)
            cells.append((k, n, 100 * p, 100 * lo, 100 * hi))
        name = m.replace("gpt-", "GPT-")
        r = (f"{name} & "
             + " & ".join(f"{c[2]:.0f}\\% \\tiny[{c[3]:.0f},{c[4]:.0f}]" for c in cells)
             + " \\\\")
        rows.append(r)
    return "\n".join(rows), models


def open_table(seen):
    models = sorted({k[1] for k in seen if k[0] in RACE_TASKS})
    rows = []
    for m in models:
        eo_k = dv_k = n = races = caught = 0
        for t in RACE_TASKS:
            eo = seen.get((t, m, "execute_once", 0))
            dv = seen.get((t, m, "differential_verify", 0))
            if eo:
                n += 1
                eo_k += int(bool(eo.get("robust")))
                if _is_race(eo.get("sweep", "")) and not eo.get("robust"):
                    races += 1
                    if dv and dv.get("robust"):
                        caught += 1
            if dv:
                dv_k += int(bool(dv.get("robust")))
        eo_p = 100 * eo_k / n if n else 0
        dv_p = 100 * dv_k / n if n else 0
        rows.append(f"{_pretty(m)} & {eo_p:.0f}\\% & {dv_p:.0f}\\% & {races} & {caught} \\\\")
    return "\n".join(rows), models


def main():
    fr = _load(FRONTIER)
    op = _load(OPEN)
    fr_body, fr_models = frontier_table(fr)
    op_body, op_models = open_table(op)
    (ROOT / "paper" / "table_frontier.tex").write_text(fr_body + "\n", encoding="utf-8")
    (ROOT / "paper" / "table_open.tex").write_text(op_body + "\n", encoding="utf-8")
    print(f"wrote paper/table_frontier.tex ({len(fr_models)} models)")
    print(fr_body)
    print(f"\nwrote paper/table_open.tex ({len(op_models)} open models)")
    print(op_body)


if __name__ == "__main__":
    main()
