"""Join dynamic scaling (results/scaling.jsonl) with static idiom analysis
(results/oversync.jsonl) and report the 'correct but not parallel' evidence
honestly, separating genuine over-synchronization from memory-bandwidth limits.

Parallel efficiency E(8) = S(8)/8. On an 8-core box:
  E >= 0.5 (S8 >= 4)  -- scales well
  0.25<=E<0.5         -- modest (often memory-bound: axpy, transform, streaming)
  E < 0.25 (S8 < 2)   -- essentially not parallel on 8 cores
  S8 < 1              -- PATHOLOGICAL: parallelism makes it slower (lock contention)

Memory-bound families (axpy, transform, gemv streaming) are expected to be
bandwidth-limited, so we flag them separately from compute-bound families
(histogram, reduce, search, graph) where low S8 implies a synchronization choice.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAL = ROOT / "results" / "scaling.jsonl"
OSYNC = ROOT / "results" / "oversync.jsonl"

# families whose arithmetic intensity is low -> speedup capped by memory bandwidth
# (axpy/transform stream memory; graph highest-degree is a memory-bound degree scan)
MEM_BOUND = {"dense_la", "transform", "graph"}


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    scal = _load(SCAL)
    osync = {(r["task"], r["model"]): r for r in _load(OSYNC)}

    rows = []
    for s in scal:
        if not s.get("self_speedup8"):
            continue
        key = (s["task"], s["model"])
        idiom = osync.get(key, {}).get("primary", "?")
        rows.append({**s, "idiom": idiom, "eff8": s["self_speedup8"] / 8,
                     "mem_bound": s["type"] in MEM_BOUND})

    n = len(rows)
    print(f"{n} robust programs with timing\n")

    # headline distribution
    patho = [r for r in rows if r["self_speedup8"] < 1.0]
    flat = [r for r in rows if r["self_speedup8"] < 2.0]
    flat_cb = [r for r in rows if r["self_speedup8"] < 2.0 and not r["mem_bound"]]
    good = [r for r in rows if r["self_speedup8"] >= 4.0]
    print(f"  pathological  (S8<1, slower on 8 threads): {len(patho):2d}  ({100*len(patho)/n:.0f}%)")
    print(f"  not-parallel  (S8<2 on 8 cores)          : {len(flat):2d}  ({100*len(flat)/n:.0f}%)")
    print(f"    of which compute-bound (not mem-limited): {len(flat_cb):2d}  ({100*len(flat_cb)/n:.0f}%)")
    print(f"  scales well   (S8>=4)                    : {len(good):2d}  ({100*len(good)/n:.0f}%)")

    print("\nmean self-speedup(8) by primary idiom:")
    byid = defaultdict(list)
    for r in rows:
        byid[r["idiom"]].append(r["self_speedup8"])
    for idm, vals in sorted(byid.items(), key=lambda kv: sum(kv[1])/len(kv[1])):
        import statistics
        print(f"  {idm:14s} n={len(vals):2d}  mean S8={statistics.mean(vals):.2f}  "
              f"min={min(vals):.2f}  max={max(vals):.2f}")

    print("\ncompute-bound programs that DON'T scale (S8<2, family expects scaling):")
    for r in sorted(flat_cb, key=lambda r: r["self_speedup8"]):
        print(f"  {r['task']:32s} {r['model']:12s} S8={r['self_speedup8']:.2f} "
              f"E8={r['eff8']:.2f} idiom={r['idiom']}")

    print("\npathological cases (parallelism actively hurts):")
    for r in sorted(patho, key=lambda r: r["self_speedup8"]):
        print(f"  {r['task']:32s} {r['model']:12s} S8={r['self_speedup8']:.2f} idiom={r['idiom']}")

    # emit LaTeX macros
    macros = [
        rf"\newcommand{{\Ntimed}}{{{n}}}",
        rf"\newcommand{{\NpathScale}}{{{len(patho)}}}",
        rf"\newcommand{{\NflatScale}}{{{len(flat)}}}",
        rf"\newcommand{{\PctFlatScale}}{{{100*len(flat)/n:.0f}\%}}",
        rf"\newcommand{{\NflatCompute}}{{{len(flat_cb)}}}",
        rf"\newcommand{{\NgoodScale}}{{{len(good)}}}",
    ]
    # critical-idiom mean speedup vs reduction
    import statistics
    crit = [r["self_speedup8"] for r in rows if r["idiom"] == "critical"]
    red = [r["self_speedup8"] for r in rows if r["idiom"] == "reduction"]
    if crit:
        macros.append(rf"\newcommand{{\MeanSpeedupCritical}}{{{statistics.mean(crit):.1f}}}")
    if red:
        macros.append(rf"\newcommand{{\MeanSpeedupReduction}}{{{statistics.mean(red):.1f}}}")
    (ROOT / "results" / "scaling_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (ROOT / "paper" / "scaling_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"\nwrote results/ + paper/ scaling_numbers.tex ({len(macros)} macros)")


if __name__ == "__main__":
    main()
