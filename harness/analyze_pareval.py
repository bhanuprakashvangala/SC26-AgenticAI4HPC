"""Analyze the agentic ParEval study: turn results/pareval_runs.jsonl into the
paper's headline ablation (single_shot vs execute_once vs differential_verify),
with Wilson 95% CIs, per-model and per-type breakdowns, a repair analysis, and a
bar-chart figure. Emits results/pareval_numbers.tex for the paper. No model or
network access needed -- purely reads the logged results.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "pareval_runs.jsonl"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COND_ORDER = ["single_shot", "execute_once", "differential_verify"]
COND_LABEL = {"single_shot": "L0 single-shot",
              "execute_once": "L1 execute-once",
              "differential_verify": "L2 differential-verify"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0.0, c-h), min(1.0, c+h))


def load():
    rows = []
    if RESULTS.exists():
        for ln in RESULTS.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    # dedup: keep last per (task,model,condition,trial)
    seen = {}
    for r in rows:
        seen[(r["task"], r["model"], r["condition"], r.get("trial", 0))] = r
    return list(seen.values())


def rate_by(rows, cond):
    sub = [r for r in rows if r["condition"] == cond]
    n = len(sub)
    k = sum(1 for r in sub if r.get("robust"))
    return k, n, wilson(k, n)


def main():
    rows = load()
    if not rows:
        print("no results yet in", RESULTS)
        return
    print(f"loaded {len(rows)} runs across "
          f"{len({r['task'] for r in rows})} tasks, "
          f"{len({r['model'] for r in rows})} models\n")

    # ---- headline ablation ----
    print("== ABLATION (robust-correct rate by condition) ==")
    macros = []
    cond_rate = {}
    for cond in COND_ORDER:
        k, n, (p, lo, hi) = rate_by(rows, cond)
        cond_rate[cond] = (k, n, p, lo, hi)
        if n:
            print(f"  {COND_LABEL[cond]:24s} {k}/{n} = {100*p:5.1f}%  "
                  f"(95% CI [{100*lo:.1f}, {100*hi:.1f}])")
    # macros
    key = {"single_shot": "Lzero", "execute_once": "Lone", "differential_verify": "Ltwo"}
    for cond, kk in key.items():
        if cond in cond_rate and cond_rate[cond][1]:
            k, n, p, lo, hi = cond_rate[cond]
            macros.append(f"\\newcommand{{\\{kk}rate}}{{{100*p:.1f}\\%}}")
            macros.append(f"\\newcommand{{\\{kk}ci}}{{[{100*lo:.1f}, {100*hi:.1f}]}}")
    if "single_shot" in cond_rate and "differential_verify" in cond_rate:
        gain = 100*(cond_rate["differential_verify"][2] - cond_rate["single_shot"][2])
        macros.append(f"\\newcommand{{\\LtwoGain}}{{{gain:.1f}\\%}}")

    # ---- repair analysis: of cells failing at single_shot, how many each condition rescues ----
    # match by (task, model, trial) across conditions
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["task"], r["model"], r.get("trial", 0))][r["condition"]] = r.get("robust")
    paired = [c for c in by_cell.values() if "single_shot" in c]
    fail_ss = [c for c in paired if not c.get("single_shot")]
    eo_fix = sum(1 for c in fail_ss if c.get("execute_once"))
    dv_fix = sum(1 for c in fail_ss if c.get("differential_verify"))
    if fail_ss:
        print(f"\n== REPAIR (of {len(fail_ss)} cells NOT robust single-shot) ==")
        print(f"  execute-once rescued:        {eo_fix}/{len(fail_ss)}")
        print(f"  differential-verify rescued: {dv_fix}/{len(fail_ss)}")
    else:
        print("\n== REPAIR == no cells failed single-shot (nothing to rescue)")
    # Always emit these macros (even at 0) so pareval_numbers.tex is self-contained:
    # once present it fully replaces pareval_defaults.tex, so every macro the paper
    # references must be defined here regardless of whether any cell failed.
    macros.append(f"\\newcommand{{\\NfailSingleShot}}{{{len(fail_ss)}}}")
    macros.append(f"\\newcommand{{\\EoRescued}}{{{eo_fix}}}")
    macros.append(f"\\newcommand{{\\DvRescued}}{{{dv_fix}}}")

    # ---- did differential verification ever change the outcome vs execute-once? ----
    # For each (task, model, trial) with both L1 and L2, count where the final
    # robustness verdict DIFFERS. This is the direct test of whether the extra
    # multi-thread signal mattered; N=0 means it never did on this set.
    paired_ld = [c for c in by_cell.values()
                 if "execute_once" in c and "differential_verify" in c]
    ndiffer = sum(1 for c in paired_ld
                  if bool(c.get("execute_once")) != bool(c.get("differential_verify")))
    dv_only = sum(1 for c in paired_ld
                  if c.get("differential_verify") and not c.get("execute_once"))
    print(f"\n== DIFFERENTIAL vs EXECUTE-ONCE (paired, n={len(paired_ld)}) ==")
    print(f"  cells where final verdict differs: {ndiffer}"
          f"  (differential-only rescues: {dv_only})")
    macros.append(f"\\newcommand{{\\Npairs}}{{{len(paired_ld)}}}")
    macros.append(f"\\newcommand{{\\Ndiffer}}{{{ndiffer}}}")
    macros.append(f"\\newcommand{{\\DvOnlyRescued}}{{{dv_only}}}")
    if "execute_once" in cond_rate and "differential_verify" in cond_rate:
        d21 = 100*(cond_rate["differential_verify"][2] - cond_rate["execute_once"][2])
        macros.append(f"\\newcommand{{\\LtwoOverLone}}{{{d21:+.1f}\\%}}")

    # ---- per-model ----
    print("\n== BY MODEL (differential-verify robust rate) ==")
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m and r["condition"] == "differential_verify"]
        if sub:
            k = sum(1 for r in sub if r.get("robust"))
            print(f"  {m:14s} {k}/{len(sub)} = {100*k/len(sub):.1f}%")

    # ---- per problem type ----
    print("\n== BY PROBLEM TYPE (robust rate, all conditions) ==")
    _typemap = {"histogram": "Hist", "reduce": "Reduce", "scan": "Scan",
                "search": "Search", "graph": "Graph"}
    for t in sorted({r["type"] for r in rows if "type" in r}):
        sub = [r for r in rows if r.get("type") == t]
        k = sum(1 for r in sub if r.get("robust"))
        print(f"  {t:12s} {k}/{len(sub)} = {100*k/len(sub):.1f}%")
        if t in _typemap:
            macros.append(f"\\newcommand{{\\Robust{_typemap[t]}}}{{{k}/{len(sub)}}}")

    macros.append(f"\\newcommand{{\\Ntasks}}{{{len({r['task'] for r in rows})}}}")
    macros.append(f"\\newcommand{{\\Nmodels}}{{{len({r['model'] for r in rows})}}}")
    macros.append(f"\\newcommand{{\\Nruns}}{{{len(rows)}}}")
    (ROOT / "results" / "pareval_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"\nwrote results/pareval_numbers.tex ({len(macros)} macros)")

    # ---- figure: robust rate by condition ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conds = [c for c in COND_ORDER if c in cond_rate and cond_rate[c][1]]
        ps = [100*cond_rate[c][2] for c in conds]
        errs = [[100*(cond_rate[c][2]-cond_rate[c][3]) for c in conds],
                [100*(cond_rate[c][4]-cond_rate[c][2]) for c in conds]]
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#C42B1C", "#9D5D00", "#0F7B0F"]
        ax.bar(range(len(conds)), ps, yerr=errs, capsize=6,
               color=[colors[COND_ORDER.index(c)] for c in conds])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=9)
        ax.set_ylabel("Configuration-robust correctness (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Agentic tool-ablation: robust correctness by tool level")
        for i, p in enumerate(ps):
            ax.text(i, p + 2, f"{p:.0f}%", ha="center", fontsize=10, weight="bold")
        fig.tight_layout()
        fig.savefig(FIGDIR / "pareval_ablation.pdf", bbox_inches="tight")
        fig.savefig(FIGDIR / "pareval_ablation.png", dpi=140, bbox_inches="tight")
        print("wrote results/figures/pareval_ablation.{pdf,png}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
