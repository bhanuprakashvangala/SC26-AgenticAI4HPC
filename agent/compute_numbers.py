"""
compute_numbers.py  --  derive every paper number the FROZEN data actually supports, and say
plainly which macros need an experiment we did not run (so nothing is fabricated).

Reads results/{scaling,pareval_runs,sanitizer}.jsonl + the generation sources via diag_core.
Prints a report; with --write, emits paper_agentic/numbers_filled.tex with the derived macros.

Honest scoping, baked in:
  * E1 serial projection  -> STATIC (source transform); labelled as such.
  * E3 signal collapse     -> the study logged 1 sample per (task,model,condition), not k-sampled
                              groups, so the dead-group rate is a PREDICTION from the measured
                              per-task pass rates: P(zero R_corr variance | k) = E_t[p_t^k+(1-p_t)^k].
  * E6 multi-architecture  -> single verification backend (NArch=1); the cross-arch rank-instability
                              macros are NOT derivable and are left for the reader to fill or drop.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_core as DC

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NMAX = 8


def load(name):
    rows = []
    for line in open(os.path.join(R, "results", name), encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def robust(r):
    return str(r.get("robust")).lower() == "true" or r.get("verdict") == "ROBUST_CORRECT"


def main(write=False):
    sess = DC.Session(verbose=False)
    progs = sess.programs
    sp = [p["self_speedup8"] for p in progs if p["self_speedup8"] is not None]
    eff = [p["parallel_efficiency8"] for p in progs if p["parallel_efficiency8"] is not None]
    n = len(sp)
    runs = load("pareval_runs.jsonl")
    ss = [r for r in runs if r["condition"] == "single_shot"]
    models = sorted({r["model"] for r in runs})
    tasks = sorted({r["task"] for r in runs})

    M = {}  # macro -> (value, provenance)

    # ---- scale ----
    M["NTasks"] = (len(tasks), "distinct ParEval tasks in the pool")
    M["NModels"] = (len(models), "frontier models: " + ", ".join(models))
    M["NRuns"] = (len(runs), "logged (task,model,condition) runs")
    M["NAccepted"] = (n, "accepted + timed programs (scaling.jsonl)")
    M["NThreadsMax"] = (NMAX, "top of the thread sweep {1,2,4,8}")

    # ---- E1 serial projection (static) ----
    proj = [p["serial_projection"] for p in progs if p.get("serial_projection")]
    still = sum(1 for x in proj if x["predicted_still_correct"])
    M["SPDen"] = (len(proj), "programs with source available for projection")
    M["SPNum"] = (still, "predicted still-correct after deleting every #pragma omp")
    M["SPRate"] = (round(100 * still / len(proj), 1), "% (STATIC projection)")
    M["SPRewardDeltaCorr"] = ("0.00", "exact: R_corr invariant to a schedule-only transform")
    M["SPGateFloor"] = (round(1.0 / NMAX, 3), "gated reward of a serial program (S=1 => 1/n)")

    # ---- E2 argmax contamination ----
    d = sess.scaling_distribution()
    M["SpeedMin"] = (d["self_speedup8_min"], "min self-speedup @8 among accepted")
    M["SpeedMax"] = (d["self_speedup8_max"], "max self-speedup @8 among accepted")
    M["SpeedMedian"] = (d["self_speedup8_median"], "median")
    M["BelowSerialRate"] = (d["below_serial_pct"], "% accepted with S8<1 (slower than serial)")
    M["BelowTwoXRate"] = (d["below_2x_pct"], "% accepted with S8<2")
    M["ContamRate"] = (d["efficiency_below_0.25_pct"], "% accepted with capped efficiency <0.25")
    ws = d["widest_within_task_spread"]
    M["WorstTaskRange"] = ("%.2f--%.2f" % (ws["min"], ws["max"]), "widest within-task spread (%s)" % ws["task"])

    # ---- E7 correctness saturation (the premise) ----
    k = sum(robust(r) for r in ss)
    M["PassRate"] = (round(100 * k / len(ss), 1), "single-shot robust-correct pass rate")
    bytask = defaultdict(list)
    for r in ss:
        bytask[r["task"]].append(robust(r))
    p_t = {t: (sum(v) / len(v)) for t, v in bytask.items() if v}
    sat = sum(1 for t, v in p_t.items() if v == 1.0)
    M["SatTaskFrac"] = (round(100 * sat / len(p_t), 0), "% tasks with single-shot pass rate == 1")

    # ---- E3 signal collapse : PREDICTION from measured per-task pass rates ----
    # For a group of k i.i.d. samples of a task with pass prob p, P(zero R_corr variance)=p^k+(1-p)^k.
    pv = list(p_t.values())
    for kk in (4, 8, 16):
        dead = st.mean([p ** kk + (1 - p) ** kk for p in pv])
        M["DeadGroupCorr%s" % {4: "Four", 8: "Eight", 16: "Sixteen"}[kk]] = (
            round(100 * dead, 1), "PREDICTED %% zero-variance groups @k=%d under R_corr (from per-task p_t)" % kk)
    # under R_gate = 1[c].f with f continuous, a group is dead only if ALL k are incorrect
    # (Prop. signal-collapse): probability at most E_t[(1-p_t)^k].
    for kk in (4, 8, 16):
        dead_g = st.mean([(1 - p) ** kk for p in pv])
        M["DeadGroupGate%s" % {4: "Four", 8: "Eight", 16: "Sixteen"}[kk]] = (
            round(100 * dead_g, 1), "PREDICTED %% zero-variance groups @k=%d under R_gate (all-incorrect only)" % kk)

    # ---- E4 best-of-N bake-off (selection component only; labelled a proxy in-text) ----
    # group timed programs by task; correctness-only selection is indifferent among the correct
    # (expected = mean speedup); the gate selects max capped-efficiency (= max speedup at fixed n).
    tg = defaultdict(list)
    for p in progs:
        if p["self_speedup8"] is not None:
            tg[p["task"]].append(p["self_speedup8"])
    corr_sel = st.mean([st.mean(v) for v in tg.values()])          # expected pick under R_corr
    gate_sel = st.mean([max(v) for v in tg.values()])              # pick under R_gate == oracle here
    M["BoNCorr"] = (round(corr_sel, 2), "realized mean S8, correctness-only selection (proxy)")
    M["BoNGate"] = (round(gate_sel, 2), "realized mean S8, efficiency-gated selection (proxy)")
    M["BoNOracle"] = (round(gate_sel, 2), "oracle-best S8 per task")
    M["BoNGain"] = (round(100 * (gate_sel - corr_sel) / corr_sel, 0),
                    "% more speedup the gate recovers from the same samples")
    M["PoolsDiffering"] = (sum(1 for v in tg.values() if len(v) >= 2 and max(v) != st.mean(v)),
                           "tasks (>=2 candidates) where the two rewards pick differently")

    # ---- report ----
    print("=" * 78)
    print("DERIVED PAPER NUMBERS  (frozen data: %d programs, %d runs, %d models, %d tasks)"
          % (n, len(runs), len(models), len(tasks)))
    print("=" * 78)
    for name, (val, prov) in M.items():
        print("  \\%-22s = %-14s  %% %s" % (name, val, prov))
    print("-" * 78)
    print("NEEDS AN EXPERIMENT WE DID NOT RUN (leave for the reader / drop from claims):")
    for name, why in [
        ("NHacks/HackCorrPassRate/HackGatePassRate/HackGateSurvivors",
         "the ParallelGate adversarial suite -- author + score separately (E5)"),
        ("CVMedian/NoiseFloor/RewardResolution",
         "per-rep timings not logged (scaling stored min-over-reps only)"),
        ("RankInstability/NArch/ArchA/ArchB/CompilerList",
         "single verification backend -- no cross-architecture run (E6)"),
        ("Regret*/Rho*", "optional selection-quality stats; derivable but omitted unless needed"),
    ]:
        print("  - %-58s %s" % (name, why))
    print("=" * 78)

    if write:
        out = os.path.join(R, "paper_agentic")
        os.makedirs(out, exist_ok=True)
        fp = os.path.join(out, "numbers_filled.tex")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("%% Auto-derived from the frozen run pool by agent/compute_numbers.py.\n")
            fh.write("%% Every value below is a real measurement or a clearly-labelled prediction.\n\n")
            # The prose appends \% itself (e.g. \PassRate\%), so every macro is a BARE value.
            for name, (val, prov) in M.items():
                fh.write("\\newcommand{\\%s}{%s}%% %s\n" % (name, val, prov))
        print("wrote", fp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    main(write=a.write)
