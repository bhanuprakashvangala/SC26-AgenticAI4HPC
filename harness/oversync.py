"""Static over-synchronization analysis of the configuration-robust programs.

Complements the dynamic scaling measurement: classifies HOW each robust program
achieves race-freedom, to test the hypothesis that models buy safety with the
safest-available idiom rather than the scalable one.

Buckets (per robust final program):
  reduction   -- uses `reduction(...)` clause: the scalable idiom
  atomic      -- uses `#pragma omp atomic`: scales but slower than reduction
  critical    -- uses `#pragma omp critical`: serializes; robust but ~unscalable
  single/master with a serial loop inside a parallel region
  serial_merge-- a post-loop sequential combine of per-thread partials
  plain       -- a bare `parallel for` with no shared-write sync (fine if no shared write)
  no_omp      -- no OpenMP pragmas at all: serial program that trivially passes R(G)
A program flagged critical/single/no_omp is a candidate "robust but not parallel".
"""
from __future__ import annotations
import json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "logs" / "pareval" / "transcripts"
OUT = ROOT / "results" / "oversync.jsonl"


def classify(code: str):
    c = code
    has_parallel = bool(re.search(r"#pragma\s+omp\s+parallel", c))
    has_for = bool(re.search(r"#pragma\s+omp\s+(parallel\s+)?for", c))
    has_reduction = bool(re.search(r"reduction\s*\(", c))
    has_atomic = bool(re.search(r"#pragma\s+omp\s+atomic", c))
    has_critical = bool(re.search(r"#pragma\s+omp\s+critical", c))
    has_single = bool(re.search(r"#pragma\s+omp\s+(single|master)", c))
    # a critical INSIDE a for-loop body is the serializing anti-pattern
    crit_in_loop = bool(re.search(r"for\s*\([^)]*\)\s*\{[^}]*#pragma\s+omp\s+critical", c, re.S)) \
        or (has_critical and has_for)
    flags = []
    if not has_parallel and not has_for:
        return "no_omp", ["no OpenMP pragmas"]
    if has_reduction:
        flags.append("reduction")
    if has_atomic:
        flags.append("atomic")
    if has_critical:
        flags.append("critical")
    if has_single:
        flags.append("single/master")
    # primary bucket = the least-scalable idiom present
    if has_critical:
        primary = "critical"
    elif has_single:
        primary = "single/master"
    elif has_atomic:
        primary = "atomic"
    elif has_reduction:
        primary = "reduction"
    else:
        primary = "plain_for"
    return primary, flags


def main():
    rows = []
    for f in TR.glob("*__differential_verify__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not (d.get("robust") and d.get("final_code")):
            continue
        primary, flags = classify(d["final_code"])
        rows.append({"task": d["task"], "type": d["type"], "model": d["model"],
                     "primary": primary, "flags": flags})
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    print(f"{len(rows)} robust programs classified\n")
    by_primary = Counter(r["primary"] for r in rows)
    for k, n in by_primary.most_common():
        print(f"  {k:16s} {n:3d}  ({100*n/len(rows):.0f}%)")
    print("\nby type x primary idiom:")
    tp = defaultdict(Counter)
    for r in rows:
        tp[r["type"]][r["primary"]] += 1
    for t in sorted(tp):
        parts = ", ".join(f"{k}:{v}" for k, v in tp[t].most_common())
        print(f"  {t:10s} {parts}")
    # the headline: fraction using a serializing idiom (critical/single/no_omp)
    unscal = sum(1 for r in rows if r["primary"] in ("critical", "single/master", "no_omp"))
    print(f"\nrobust-but-serializing-idiom: {unscal}/{len(rows)} "
          f"({100*unscal/len(rows):.0f}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
