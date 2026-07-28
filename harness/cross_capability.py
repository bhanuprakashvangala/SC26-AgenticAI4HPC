"""Cross-capability analysis: contrast the frontier panel (results/pareval_runs.jsonl)
with the open-weights panel (results/pareval_open.jsonl) on the SHARED race-prone
task set, and emit the positive-result figure + macros.

The paper's null says: for frontier models, execute-once (L1) and differential
verification (L2) reach an identical outcome -- 0 silent races. This module tests
the complementary hypothesis: weaker open models DO emit the silent-race failure
mode, so L2 rescues cells that L1 certifies as correct. That is the differential
verifier's value made visible, on real model output rather than injected races.

Definitions (all from the logged full-thread-sweep score of each final program):
  silent race  : final program passes at 1 thread but fails at a higher count.
  L1 certified : the execute_once-condition final program is scored robust=False
                 yet passed the 1-thread check the agent was allowed to use.
  L2 rescued   : for the same task, the differential-condition final is robust.

Outputs:
  results/figures/fig_cross_capability.{pdf,png}
  results/cross_numbers.tex  (+ mirrored into paper/)
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "results" / "pareval_runs.jsonl"
OPEN = ROOT / "results" / "pareval_open.jsonl"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# the race-prone task set both panels share
RACE_TASKS = ["21_histogram_bin_0-100", "22_histogram_count_quadrants",
              "26_reduce_product_of_inverses", "29_reduce_sum_of_min_of_pairs",
              "33_scan_reverse_prefix_sum", "17_graph_highest_degree"]
COND = ["single_shot", "execute_once", "differential_verify"]
C_L1, C_L2 = "#E08A00", "#1E6F3C"


def _load(path):
    seen = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                seen[(r["task"], r["model"], r["condition"], r.get("trial", 0))] = r
    return seen


def _pretty(m: str) -> str:
    """Compact model label for axes."""
    return m.replace("gpt-", "").replace("-small", "-sm")


def _group_bracket(ax, x0, x1, label, color):
    """Draw a group-span bracket below the tick labels with a centered label,
    using axis-fraction y so it never collides with bars or the title."""
    y = -0.30
    ax.plot([x0, x1], [y, y], color=color, lw=1.6, clip_on=False,
            transform=ax.get_xaxis_transform())
    for xe in (x0, x1):
        ax.plot([xe, xe], [y, y + 0.02], color=color, lw=1.6, clip_on=False,
                transform=ax.get_xaxis_transform())
    ax.text((x0 + x1) / 2, y - 0.05, label, ha="center", va="top", fontsize=8.6,
            style="italic", color=color, clip_on=False,
            transform=ax.get_xaxis_transform())


def is_race(sweep: str) -> bool:
    m = {int(a): (int(b), int(c))
         for a, b, c in re.findall(r"t(\d+)=(\d+)/(\d+)", sweep or "")}
    if not m:
        return False
    p1 = m.get(1, (0, 0))
    hi = [m[t] for t in m if t > 1]
    return p1[1] > 0 and p1[0] == p1[1] and any(b < tot for b, tot in hi)


def panel_stats(seen, models, tasks):
    """Per model: robust counts per condition + races certified by L1 that L2 catches."""
    out = {}
    for mdl in models:
        rec = {c: [0, 0] for c in COND}          # [robust, total]
        l1_races = 0                              # execute_once final is a silent race
        l2_rescued = 0                            # same task robust under differential
        for t in tasks:
            for c in COND:
                r = seen.get((t, mdl, c, 0))
                if not r:
                    continue
                rec[c][1] += 1
                rec[c][0] += int(bool(r.get("robust")))
            eo = seen.get((t, mdl, "execute_once", 0))
            dv = seen.get((t, mdl, "differential_verify", 0))
            if eo and is_race(eo.get("sweep", "")) and not eo.get("robust"):
                l1_races += 1
                if dv and dv.get("robust"):
                    l2_rescued += 1
        out[mdl] = {"rec": rec, "l1_races": l1_races, "l2_rescued": l2_rescued}
    return out


def main():
    fr = _load(FRONTIER)
    op = _load(OPEN)
    fr_models = sorted({k[1] for k in fr if k[0] in RACE_TASKS})
    op_models = sorted({k[1] for k in op if k[0] in RACE_TASKS})
    print("frontier models:", fr_models)
    print("open models    :", op_models)

    fr_stat = panel_stats(fr, fr_models, RACE_TASKS)
    op_stat = panel_stats(op, op_models, RACE_TASKS)

    fr_races = sum(s["l1_races"] for s in fr_stat.values())
    op_races = sum(s["l1_races"] for s in op_stat.values())
    op_rescued = sum(s["l2_rescued"] for s in op_stat.values())
    print(f"\nfrontier silent races certified by L1: {fr_races}")
    print(f"open     silent races certified by L1: {op_races}  (L2 rescued {op_rescued})")
    for m, s in op_stat.items():
        print(f"  {m:14s} L1_races={s['l1_races']} L2_rescued={s['l2_rescued']} "
              f"L1_robust={s['rec']['execute_once']} L2_robust={s['rec']['differential_verify']}")

    # ---------------- figure: two clean panels, no text collisions ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from harness.figstyle import (apply_style, wilson, despine, C_L1, C_L2,
                                  C_RACE, MUTE, INK)
    apply_style(plt)

    order = [(m, fr_stat[m], "frontier") for m in fr_models] + \
            [(m, op_stat[m], "open") for m in op_models]
    labels = [_pretty(m) for m, _, _ in order]
    n_fr = len(fr_models)

    def _rate(s, c):
        k, n = s["rec"][c]
        return 100 * k / n if n else 0.0

    def _ci(s, c):
        k, n = s["rec"][c]
        p, lo, hi = wilson(k, n)
        return 100 * p, 100 * (p - lo), 100 * (hi - p)

    l1 = [_rate(s, "execute_once") for _, s, _ in order]
    l2 = [_rate(s, "differential_verify") for _, s, _ in order]
    rescued = [s["l2_rescued"] for _, s, _ in order]

    x = np.arange(len(labels))
    w = 0.36
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [2.15, 1.0]})

    # ---- Panel A: robust correctness, L1 vs L2, per model ----
    bA1 = axA.bar(x - w/2, l1, w, label="L1 execute-once", color=C_L1,
                  edgecolor="white", lw=1.0, zorder=3)
    bA2 = axA.bar(x + w/2, l2, w, label="L2 differential", color=C_L2,
                  edgecolor="white", lw=1.0, zorder=3)
    # highlight ONLY genuine race rescues (L1 shipped a silent race, L2 caught it)
    for i in range(len(order)):
        if rescued[i] > 0:
            axA.annotate("", xy=(x[i] + w/2, l2[i] - 0.5), xytext=(x[i] + w/2, l1[i]),
                         arrowprops=dict(arrowstyle="-|>", color=C_RACE, lw=2.4))
            axA.text(x[i] + w/2, l2[i] + 3.5,
                     "verified race\nL1 shipped, L2 caught", fontsize=8.2, color=C_RACE,
                     weight="bold", ha="center", va="bottom", linespacing=1.15)
    despine(axA)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)
    axA.set_ylabel("robust correct on shared\nrace-prone tasks (%)")
    axA.set_ylim(0, 100)
    axA.set_yticks([0, 25, 50, 75, 100])
    axA.legend(loc="upper left", fontsize=9.5, ncol=1)
    axA.set_title("(a)  L1 and L2 agree \u2014 except where a model ships a real race",
                  fontsize=11)
    # frontier | open group brackets BELOW the tick labels (no collision)
    _group_bracket(axA, -0.5 + 0.15, n_fr - 1 + 0.35, "frontier (Azure)", MUTE)
    if op_models:
        _group_bracket(axA, n_fr - 0.35, len(labels) - 1 + 0.35,
                       "open-weights (NRP)", MUTE)
    axA.axvline(n_fr - 0.5, color="0.55", ls="--", lw=1.1, zorder=1)

    # ---- Panel B: the thesis in one bar per model -- silent races L2 caught ----
    from harness.figstyle import lighten
    colors = [C_RACE if r else "#D5D8DC" for r in rescued]
    bB = axB.bar(x, rescued, 0.62, color=colors, edgecolor="white", lw=1.0, zorder=3)
    for i, r in enumerate(rescued):
        if r:
            axB.text(x[i], r + 0.04, str(r), ha="center", va="bottom",
                     fontsize=12, weight="bold", color=C_RACE)
    despine(axB)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)
    axB.set_ylabel("silent races L1 shipped\nthat L2 caught")
    top = max(rescued + [1])
    axB.set_ylim(0, top + 0.55)
    axB.set_yticks(range(top + 1))
    axB.axvline(n_fr - 0.5, color=MUTE, ls=(0, (3, 3)), lw=1.1, zorder=1)
    axB.set_title("(b)  The verifier's payoff\nrises as capability falls", fontsize=10.5)

    fig.suptitle("Differential verification catches the silent races only weaker models emit",
                 fontsize=13, weight="bold", y=1.04)
    fig.subplots_adjust(wspace=0.46, bottom=0.28, top=0.82)
    fig.savefig(FIG / "fig_cross_capability.pdf")
    fig.savefig(FIG / "fig_cross_capability.png", dpi=150)
    plt.close(fig)
    print("  wrote fig_cross_capability")

    # ---------------- macros ----------------
    macros = [
        rf"\newcommand{{\NopenModels}}{{{len(op_models)}}}",
        rf"\newcommand{{\NfrontierRaces}}{{{fr_races}}}",
        rf"\newcommand{{\NopenRaces}}{{{op_races}}}",
        rf"\newcommand{{\NopenRescued}}{{{op_rescued}}}",
        rf"\newcommand{{\NraceTasks}}{{{len(RACE_TASKS)}}}",
    ]
    (ROOT / "results" / "cross_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (ROOT / "paper" / "cross_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"\nwrote results/ + paper/ cross_numbers.tex ({len(macros)} macros)")


if __name__ == "__main__":
    main()
