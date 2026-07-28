"""The performance verification gate -- the second gate VG-RLPT adds.

The correctness gate (differential_verify in run_pareval_batch) answers "does it agree
with the serial reference at every thread count?" This gate answers the second
question the paper is about: "does it actually get faster with more threads?"

It compiles a candidate program against ParEval's timing driver and runs the same
thread sweep {1,2,4,8}, reusing measure_scaling's proven bench/parse path, and returns
the measured 8-thread self-speedup plus a termination decision beta:

  beta_parallel(program) = 1   iff  self_speedup(8) >= target   (the program scales)

So the gate is *guarded by correctness*: we only ever ask "is it fast?" of a program
that already cleared the correctness gate. That guard is the whole design -- it is why
optimizing this reward cannot be satisfied by fast-but-wrong code.

Runs on the Nautilus pod (VERIFY_BACKEND=nautilus / kubectl exec), same as
measure_scaling. Import and call time_program(); or run as a module for a smoke test
against a transcript.
"""
from __future__ import annotations
from dataclasses import dataclass

from harness import measure_scaling as ms


@dataclass
class PerfResult:
    compiled: bool
    valid: str | None
    speedup8: float | None
    vs_serial8: float | None
    times: dict            # {thread_count: min_wall_time}
    scales: bool           # beta_parallel: did it clear the performance gate?

    @property
    def robust_timing(self) -> bool:
        return self.compiled and self.valid == "PASS" and bool(self.speedup8)


def time_program(code: str, task: str, ptype: str, *, model: str = "agent",
                 trial: int = 0, target: float = 2.0) -> PerfResult:
    """Compile `code` for `task` and measure its thread-sweep self-speedup.

    target: the 8-thread self-speedup a program must reach to clear the gate. 2.0 is a
    deliberately modest bar -- it only asks that eight threads beat one by 2x -- yet the
    paper shows many robustly-correct programs fail even this.
    """
    cell = {"final_code": code, "task": task, "type": ptype,
            "model": model, "condition": "vg", "trial": trial}
    script = "#!/bin/bash\n" + ms._bench_block(cell) + "\necho ALLDONE\n"
    msg = ms._invoke_pod(script)
    cid = f"{task}__{model}__vg__t{trial}".replace(".", "-")
    r = ms._parse(msg).get(cid, {})
    th = r.get("threads", {})
    best = r.get("best", {})
    s8 = (th.get(1) / th.get(8)) if th.get(1) and th.get(8) else None
    vs8 = (best.get(8) / th.get(8)) if best.get(8) and th.get(8) else None
    return PerfResult(
        compiled=r.get("compiled", False),
        valid=r.get("valid"),
        speedup8=s8,
        vs_serial8=vs8,
        times=th,
        scales=bool(s8 and s8 >= target),
    )


def gate_feedback(res: PerfResult, target: float = 2.0) -> str:
    """Natural-language diagnostic the agent gets when the performance gate fails --
    the observation that drives the performance-repair step."""
    if not res.compiled:
        return "the optimized version failed to compile; revert to the last correct version and try a different parallelization."
    if res.valid and res.valid != "PASS":
        return "the optimized version is no longer correct; it must stay correct at every thread count."
    if res.speedup8 is None:
        return "could not measure a speedup; ensure the parallel region does real work at scale."
    return (f"correct, but it only reaches {res.speedup8:.2f}x self-speedup on 8 threads "
            f"(target >= {target:.1f}x). It is not exploiting the parallelism. Likely causes: "
            "a serialized critical section or per-element atomic where a reduction/private "
            "accumulator belongs, false sharing, or too little work per thread. Rewrite to "
            "scale while staying correct at every thread count.")


if __name__ == "__main__":  # smoke test against the first robust transcript on disk
    import sys
    cells = ms._load_robust("differential_verify")
    if not cells:
        print("no robust transcripts found; run the study first.")
        sys.exit(0)
    c = cells[0]
    print(f"timing {c['task']} ({c['model']}) ...")
    r = time_program(c["final_code"], c["task"], c["type"])
    print(r)
    print(gate_feedback(r))
