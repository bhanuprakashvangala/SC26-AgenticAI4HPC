"""Public entry point for ParallelRewardBench (PRB).

The current implementation originated under the historical module name
``parallel_gate``.  The paper and artifact now use the benchmark name
ParallelRewardBench; this module provides the stable public entry point while
preserving backward compatibility with the frozen analysis code.

Run offline with:

    python -m agent.parallel_reward_bench
    python -m agent.parallel_reward_bench --write
"""

from __future__ import annotations

import argparse

from .parallel_gate import HACKS, REWARDS, GENUINE, DEFEAT_FRAC, build, main

__all__ = [
    "HACKS",
    "REWARDS",
    "GENUINE",
    "DEFEAT_FRAC",
    "build",
    "main",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ParallelRewardBench on the frozen artifact data.")
    parser.add_argument("--write", action="store_true", help="write derived benchmark outputs")
    args = parser.parse_args()
    main(write=args.write)
