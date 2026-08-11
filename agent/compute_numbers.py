"""Compute reproducibility summary statistics from the frozen artifact data.

This module is intentionally offline: it reads the released JSON/JSONL files and
prints the metrics that can be derived from them. With ``--write`` it stores a
machine-readable summary in ``results/derived/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from . import diag_core as DC

ROOT = Path(DC._ROOT)
NMAX = 8


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _robust(row: dict) -> bool:
    return str(row.get("robust")).lower() == "true" or row.get("verdict") == "ROBUST_CORRECT"


def compute() -> dict:
    session = DC.Session(verbose=False)
    programs = [p for p in session.programs if p.get("self_speedup8") is not None]
    runs = _load_jsonl(ROOT / "results" / "pareval_runs.jsonl")

    speeds = [float(p["self_speedup8"]) for p in programs]
    tasks = sorted({r.get("task") for r in runs if r.get("task")})
    models = sorted({r.get("model") for r in runs if r.get("model")})

    by_task: dict[str, list[dict]] = defaultdict(list)
    for program in programs:
        by_task[program["task"]].append(program)

    widest = None
    for task, group in by_task.items():
        vals = [float(p["self_speedup8"]) for p in group]
        if not vals:
            continue
        item = {"task": task, "min": min(vals), "max": max(vals), "span": max(vals) - min(vals)}
        if widest is None or item["span"] > widest["span"]:
            widest = item

    # Correctness-only selection is indifferent among accepted candidates, so use
    # the mean self-speedup in each task pool as the expected random tie-break.
    corr_selected = st.mean(
        st.mean(float(p["self_speedup8"]) for p in group)
        for group in by_task.values() if group
    )

    # Performance-aware selection uses the trusted-reference reward.
    perf_selected_values = []
    self_selected_values = []
    for group in by_task.values():
        valid = [p for p in group if p.get("vs_serial8") is not None]
        if valid:
            chosen = max(valid, key=lambda p: min(float(p["vs_serial8"]) / NMAX, 1.0))
            perf_selected_values.append(float(chosen["self_speedup8"]))
        chosen_self = max(group, key=lambda p: float(p["self_speedup8"]))
        self_selected_values.append(float(chosen_self["self_speedup8"]))

    perf_selected = st.mean(perf_selected_values) if perf_selected_values else None
    self_selected = st.mean(self_selected_values) if self_selected_values else None

    projections = [p.get("serial_projection") for p in programs if p.get("serial_projection")]
    static_preserved = sum(1 for p in projections if p.get("predicted_still_correct"))

    single_shot = [r for r in runs if r.get("condition") == "single_shot"]
    single_shot_robust = sum(_robust(r) for r in single_shot)

    return {
        "n_tasks": len(tasks),
        "n_models": len(models),
        "models": models,
        "n_logged_runs": len(runs),
        "n_accepted_timed": len(programs),
        "max_threads": NMAX,
        "self_speedup_8": {
            "min": min(speeds),
            "max": max(speeds),
            "median": st.median(speeds),
            "below_1x_count": sum(v < 1.0 for v in speeds),
            "below_2x_count": sum(v < 2.0 for v in speeds),
            "widest_within_task": widest,
        },
        "selection_proxy": {
            "correctness_only_mean_self_speedup": corr_selected,
            "performance_aware_mean_self_speedup": perf_selected,
            "self_speedup_ablation_mean_self_speedup": self_selected,
            "performance_aware_gain_pct": (
                100.0 * (perf_selected - corr_selected) / corr_selected
                if perf_selected is not None and corr_selected else None
            ),
        },
        "serial_projection_static_check": {
            "n_with_source": len(projections),
            "predicted_correct_after_directive_removal": static_preserved,
            "note": "Static source-transform check in agent.diag_core; not a replacement for live re-execution.",
        },
        "single_shot_correctness": {
            "accepted": single_shot_robust,
            "total": len(single_shot),
            "rate": single_shot_robust / len(single_shot) if single_shot else None,
        },
    }


def print_summary(summary: dict) -> None:
    s = summary["self_speedup_8"]
    sel = summary["selection_proxy"]
    print("Frozen artifact summary")
    print("=" * 72)
    print(f"accepted + timed programs : {summary['n_accepted_timed']}")
    print(f"tasks / models             : {summary['n_tasks']} / {summary['n_models']}")
    print(f"8-thread self-speedup      : {s['min']:.2f}x -- {s['max']:.2f}x (median {s['median']:.2f}x)")
    print(f"below 2x / below 1x        : {s['below_2x_count']} / {s['below_1x_count']}")
    if s["widest_within_task"]:
        w = s["widest_within_task"]
        print(f"widest within-task range   : {w['min']:.2f}x -- {w['max']:.2f}x ({w['task']})")
    print(f"correctness-only selection : {sel['correctness_only_mean_self_speedup']:.2f}x")
    if sel["performance_aware_mean_self_speedup"] is not None:
        print(f"performance-aware selection: {sel['performance_aware_mean_self_speedup']:.2f}x")
        print(f"selection gain             : {sel['performance_aware_gain_pct']:.1f}%")
    if sel["self_speedup_ablation_mean_self_speedup"] is not None:
        print(f"self-speedup ablation      : {sel['self_speedup_ablation_mean_self_speedup']:.2f}x")
    p = summary["serial_projection_static_check"]
    print(f"static serial projection   : {p['predicted_correct_after_directive_removal']}/{p['n_with_source']} predicted value-preserving")


def main(write: bool = False) -> None:
    summary = compute()
    print_summary(summary)
    if write:
        out = ROOT / "results" / "derived"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute frozen artifact summary statistics")
    parser.add_argument("--write", action="store_true", help="write results/derived/summary.json")
    args = parser.parse_args()
    main(write=args.write)
