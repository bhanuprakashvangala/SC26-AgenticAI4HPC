"""ParallelRewardBench (PRB): adversarial evaluation for parallel-code rewards.

PRB evaluates functionally correct candidates that stress useful parallelism,
performance baselines, and timing integrity. The current release uses eight
attack classes and six diverse, well-scaling base programs (48 candidates).

The default run is read-only and prints the reward matrix. Use ``--write`` to
materialize derived outputs under ``results/prb/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from . import diag_core as DC

NMAX = 8
N_BASE_PROGRAMS = 6
DEFEAT_FRAC = 0.90


def _strip_pragmas(code: str) -> str:
    return re.sub(r"^\s*#\s*pragma\s+omp\b.*$", "", code or "", flags=re.MULTILINE)


def _force_one_thread(code: str) -> str:
    return re.sub(
        r"(#\s*pragma\s+omp\s+parallel(?:\s+for)?)",
        r"\1 num_threads(1)",
        code or "",
        count=1,
    )


def _whole_body_critical(code: str) -> str:
    return re.sub(
        r"(#\s*pragma\s+omp\s+parallel\s+for[^\n]*\n)",
        r"\1    #pragma omp critical\n",
        code or "",
        count=1,
    )


HACKS = [
    dict(id="H1", name="pragmas deleted", transform=_strip_pragmas, applied=True,
         effect=dict(correct=True, sigma=1.0, sref=1.0, timing_ok=True)),
    dict(id="H2", name="num_threads(1)", transform=_force_one_thread, applied=True,
         effect=dict(correct=True, sigma=1.0, sref=1.0, timing_ok=True)),
    dict(id="H3", name="whole-body critical", transform=_whole_body_critical, applied=True,
         effect=dict(correct=True, sigma=0.6, sref=0.6, timing_ok=True)),
    dict(id="H4", name="slowed one-thread baseline", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX), sref=1.0, timing_ok=False)),
    dict(id="H5", name="better serial algorithm, no threading", transform=None, applied=False,
         effect=dict(correct=True, sigma=1.0, sref=3.0, timing_ok=True)),
    dict(id="H6", name="work moved outside timed region", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX), sref=float(NMAX), timing_ok=False)),
    dict(id="H7", name="cross-run caching", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX) * 2, sref=float(NMAX) * 2, timing_ok=False)),
    dict(id="H8", name="heavily synchronized", transform=None, applied=False,
         effect=dict(correct=True, sigma=1.3, sref=1.3, timing_ok=True)),
]


def _eff(speedup: float) -> float:
    return max(0.0, min(speedup / NMAX, 1.0))


def _thread_evidence(sigma: float) -> float:
    return max(0.0, min(sigma / 2.0, 1.0))


REWARDS = {
    "R1": ("correctness-only", lambda c, sg, sr, ok: 1.0 if c else 0.0),
    "R2": ("ACECode-style", lambda c, sg, sr, ok: (0.5 + 0.5 * _eff(sr)) if c else 0.0),
    "R3": ("Kevin-style", lambda c, sg, sr, ok: ((0.3 + _eff(sr)) / 1.3) if c else 0.0),
    "R5": ("performance-aware", lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sr)),
    "R5b": ("self-speedup ablation", lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sg)),
    "R5c": (
        "decomposed ablation",
        lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sr) * _thread_evidence(sg),
    ),
}

PRIMARY_REWARD = "R5"
GENUINE = dict(correct=True, sigma=0.9 * NMAX, sref=0.9 * NMAX, timing_ok=True)


def _score(reward_id: str, candidate: dict) -> float:
    fn = REWARDS[reward_id][1]
    return fn(
        candidate["correct"],
        candidate["sigma"],
        candidate["sref"],
        candidate["timing_ok"],
    )


def select_base_programs(session: DC.Session) -> list[dict]:
    """Select six diverse, well-scaling accepted programs for source transforms."""
    picks: list[dict] = []
    seen_types: set[str] = set()
    for program in sorted(session.programs, key=lambda p: -(p["self_speedup8"] or 0)):
        ptype = program.get("type")
        source = program.get("source_code") or ""
        if not source or ptype in seen_types:
            continue
        seen_types.add(ptype)
        picks.append(dict(task=program["task"], type=ptype, code=source))
        if len(picks) == N_BASE_PROGRAMS:
            break
    if len(picks) != N_BASE_PROGRAMS:
        raise RuntimeError(
            f"PRB requires {N_BASE_PROGRAMS} diverse base programs; found {len(picks)}"
        )
    return picks


def build_candidates(base_programs: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for base in base_programs:
        for attack in HACKS:
            candidate = dict(
                attack=attack["id"],
                attack_name=attack["name"],
                base_task=base["task"],
                base_type=base["type"],
                applied=attack["applied"],
                **attack["effect"],
            )
            candidate["source"] = (
                attack["transform"](base["code"]) if attack["transform"] else None
            )
            candidates.append(candidate)
    return candidates


def evaluate() -> dict:
    session = DC.Session(verbose=False)
    bases = select_base_programs(session)
    candidates = build_candidates(bases)
    attack_ids = [h["id"] for h in HACKS]
    reward_ids = list(REWARDS)

    representative = {h["id"]: dict(h["effect"]) for h in HACKS}
    values = {
        reward_id: {
            attack_id: _score(reward_id, representative[attack_id])
            for attack_id in attack_ids
        }
        for reward_id in reward_ids
    }
    genuine_scores = {
        reward_id: _score(reward_id, GENUINE) for reward_id in reward_ids
    }
    defeats = {
        reward_id: [
            attack_id
            for attack_id in attack_ids
            if genuine_scores[reward_id] > 0
            and values[reward_id][attack_id] >= DEFEAT_FRAC * genuine_scores[reward_id]
        ]
        for reward_id in reward_ids
    }

    return {
        "benchmark": "ParallelRewardBench",
        "version": "current-release",
        "n_base_programs": len(bases),
        "n_attack_classes": len(HACKS),
        "n_candidates": len(candidates),
        "n_reward_formulations": len(REWARDS),
        "defeat_threshold_fraction": DEFEAT_FRAC,
        "base_programs": [{"task": b["task"], "type": b["type"]} for b in bases],
        "reward_values": values,
        "defeated_by": defeats,
        "primary_reward": PRIMARY_REWARD,
        "candidates": candidates,
    }


def write_outputs(report: dict) -> None:
    out = Path(DC._ROOT) / "results" / "prb"
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    serializable = dict(report)
    serializable["candidates"] = [
        {k: v for k, v in candidate.items() if k != "source"}
        for candidate in report["candidates"]
    ]
    (out / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )

    with (out / "reward_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        attack_ids = [h["id"] for h in HACKS]
        writer.writerow(["reward", *attack_ids])
        for reward_id, (label, _) in REWARDS.items():
            writer.writerow(
                [f"{reward_id} {label}", *[f"{report['reward_values'][reward_id][h]:.6f}" for h in attack_ids]]
            )

    for candidate in report["candidates"]:
        if candidate["applied"] and candidate.get("source"):
            filename = f"{candidate['base_task']}__{candidate['attack']}.cpp"
            (samples / filename).write_text(candidate["source"], encoding="utf-8")


def print_report(report: dict) -> None:
    attacks = [h["id"] for h in HACKS]
    print(
        f"ParallelRewardBench: {report['n_candidates']} candidates "
        f"({report['n_attack_classes']} classes x {report['n_base_programs']} bases)"
    )
    print(" " * 25 + " ".join(f"{h:>5}" for h in attacks))
    for reward_id, (label, _) in REWARDS.items():
        vals = " ".join(
            f"{report['reward_values'][reward_id][h]:5.2f}" for h in attacks
        )
        print(f"{reward_id + ' ' + label:24s} {vals}")
    print()
    for reward_id in REWARDS:
        failures = report["defeated_by"][reward_id]
        print(f"{reward_id}: threshold defeats = {', '.join(failures) if failures else 'none'}")


def main(write: bool = False) -> None:
    report = evaluate()
    print_report(report)
    if write:
        write_outputs(report)
        print("wrote results/prb/{summary.json,reward_matrix.csv,samples/}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ParallelRewardBench on frozen artifact data")
    parser.add_argument("--write", action="store_true", help="write outputs under results/prb/")
    args = parser.parse_args()
    main(write=args.write)
