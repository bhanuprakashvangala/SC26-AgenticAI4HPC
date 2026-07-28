"""The verifiable rewards at the center of the paper.

The whole argument is about *what an execution verifier rewards*. This module makes
that concrete and testable, with no model or cluster access required, so the reward
definitions can be unit-tested and reused everywhere (the VG-RLPT agent, best-of-N
selection, and GRPO training all import from here).

Three rewards, matching the paper:

  correctness_only  -- the standard, GAMEABLE reward. It is 1.0 for any program that
                       matches the serial reference across the thread sweep. A serial
                       program (no #pragma) earns a perfect 1.0, so this reward is
                       maximized by deleting the parallelism. This is the reward we
                       show an RL loop would hack.

  performance       -- parallel efficiency E(p) = speedup / p, in [0, 1]. A serial
                       program sits at 1/p (~0.125 on 8 threads); an ideally-scaling
                       program approaches 1.0. Undefined unless the program is
                       correct (you do not reward the speed of a wrong answer).

  two_axis          -- the fix. Correctness is a GATE, performance is the score behind
                       it: reward = 0 (or a small shaped correctness credit for
                       training) until the program is robustly correct, then it rises
                       with realized parallel efficiency. A serial program clears the
                       gate but scores near the floor; a fast, correct program scores
                       near the top. This is the reward VG-RLPT's two gates optimize.

`shaped=True` returns dense, partial-credit variants suitable as an RL training
signal (GRPO); `shaped=False` returns the crisp evaluation reward used for scoring
and best-of-N selection.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    p: int = 8                 # thread count the performance axis is scored at
    correct_credit: float = 0.30   # (shaped) reward floor for a correct-but-serial program
    crash_penalty: float = 0.0     # reward for code that does not compile/run
    perf_weight: float = 0.70      # (shaped) share of reward carried by the performance axis


DEFAULT = RewardConfig()


def parse_sweep(sweep: str) -> dict[int, tuple[int, int]]:
    """'t1=2/2 t2=0/2 t4=0/2 t8=0/2' -> {1:(2,2), 2:(0,2), 4:(0,2), 8:(0,2)}."""
    out: dict[int, tuple[int, int]] = {}
    for t, a, b in re.findall(r"t(\d+)=(\d+)/(\d+)", sweep or ""):
        out[int(t)] = (int(a), int(b))
    return out


def is_robust(verdict: str) -> bool:
    return verdict == "ROBUST_CORRECT"


def correctness_only(verdict: str, sweep: str = "", *, shaped: bool = False,
                     cfg: RewardConfig = DEFAULT) -> float:
    """The gameable reward. 1.0 iff robustly correct across the sweep.

    With shaped=True, award partial credit = fraction of sweep runs that passed, so an
    RL loop gets a gradient toward correctness -- but note the maximum, 1.0, is still
    reachable by a serial program, which is the whole point.
    """
    if verdict == "CRASH":
        return cfg.crash_penalty
    if is_robust(verdict):
        return 1.0
    if not shaped:
        return 0.0
    runs = parse_sweep(sweep)
    passed = sum(a for a, _ in runs.values())
    total = sum(b for _, b in runs.values())
    return (passed / total) if total else cfg.crash_penalty


def efficiency(speedup: float | None, cfg: RewardConfig = DEFAULT) -> float:
    """Parallel efficiency E = speedup / p, clamped to [0, 1]. None -> 0."""
    if not speedup or speedup <= 0:
        return 0.0
    return max(0.0, min(1.0, speedup / cfg.p))


def performance(speedup: float | None, verdict: str = "ROBUST_CORRECT",
                cfg: RewardConfig = DEFAULT) -> float:
    """Performance axis: efficiency, but only for correct programs (you do not
    reward the speed of a wrong answer)."""
    if not is_robust(verdict):
        return 0.0
    return efficiency(speedup, cfg)


def two_axis(verdict: str, speedup: float | None, sweep: str = "", *,
             shaped: bool = False, cfg: RewardConfig = DEFAULT) -> float:
    """Correctness gate x performance score -- the reward VG-RLPT optimizes.

    shaped=False (evaluation): 0 unless robust; then efficiency in [0,1].
    shaped=True  (training)  : dense. Partial correctness credit up to
                 `correct_credit`, plus `perf_weight` * efficiency once correct, so
                 the RL signal first pulls the model to correctness, then -- unlike the
                 correctness-only reward -- keeps pulling toward real parallelism.
    """
    if not shaped:
        return efficiency(speedup, cfg) if is_robust(verdict) else 0.0
    if verdict == "CRASH":
        return cfg.crash_penalty
    if not is_robust(verdict):
        # partial credit for getting closer to correct, capped below the correct floor
        return cfg.correct_credit * correctness_only(verdict, sweep, shaped=True, cfg=cfg)
    return cfg.correct_credit + cfg.perf_weight * efficiency(speedup, cfg)


# --------------------------- self-test (no deps) ---------------------------
if __name__ == "__main__":
    cfg = DEFAULT
    serial = dict(verdict="ROBUST_CORRECT", speedup=1.0, sweep="t1=2/2 t2=2/2 t4=2/2 t8=2/2")
    fast = dict(verdict="ROBUST_CORRECT", speedup=6.5, sweep="t1=2/2 t2=2/2 t4=2/2 t8=2/2")
    racy = dict(verdict="NOT_ROBUST", speedup=None, sweep="t1=2/2 t2=0/2 t4=0/2 t8=0/2")

    print("reward             serial   fast   racy")
    for name, fn in [
        ("correctness_only", lambda d: correctness_only(d["verdict"], d["sweep"])),
        ("performance      ", lambda d: performance(d["speedup"], d["verdict"])),
        ("two_axis (eval)  ", lambda d: two_axis(d["verdict"], d["speedup"], d["sweep"])),
        ("two_axis (shaped)", lambda d: two_axis(d["verdict"], d["speedup"], d["sweep"], shaped=True)),
    ]:
        print(f"{name}  {fn(serial):6.3f} {fn(fast):6.3f} {fn(racy):6.3f}")

    # the paper's claim, as assertions:
    assert correctness_only("ROBUST_CORRECT", serial["sweep"]) == 1.0        # serial aces it
    assert correctness_only(fast["verdict"]) == correctness_only(serial["verdict"])  # cannot tell apart
    assert two_axis("ROBUST_CORRECT", 6.5, fast["sweep"]) > \
           two_axis("ROBUST_CORRECT", 1.0, serial["sweep"])                  # two-axis CAN
    print("\nOK: correctness-only cannot distinguish serial from fast; two-axis can.")
