"""The 'correct but not parallel' figure: correctness is necessary but not
sufficient. Reads results/scaling.jsonl + results/oversync.jsonl and draws a
two-panel argument that the configuration-robust-correctness metric is gameable.

Panel (a): two programs the differential verifier accepted as EQUALLY robust on
the same task (histogram) -- one scales 6.5x, the other runs 3x SLOWER on 8
threads. Same verdict, opposite parallelism.
Panel (b): self-speedup(8) for every accepted program, sorted, colored by the
synchronization idiom, with the break-even (S=1) and ideal (S=8) lines. Robust
correctness spans 0.33x-7.9x -- a serial program would score identically.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAL = ROOT / "results" / "scaling.jsonl"
OSYNC = ROOT / "results" / "oversync.jsonl"
FIG = ROOT / "results" / "figures"
THREADS = [1, 2, 4, 8]


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from harness.figstyle import (apply_style, despine, C_L2, C_RACE, C_L1,
                                  MUTE, INK, lighten)
    apply_style(plt)

    scal = _load(SCAL)
    osync = {(r["task"], r["model"]): r.get("primary", "?") for r in _load(OSYNC)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.35]})

    # ---- Panel A: two 'equally robust' programs, opposite scaling ----
    def curve(task, model):
        for r in scal:
            if r["task"] == task and r["model"] == model and r.get("times"):
                t = {int(k): v for k, v in r["times"].items()}
                base = t.get(1)
                return [t.get(p) and base / t[p] for p in THREADS]
        return None

    good = curve("21_histogram_bin_0-100", "gpt-5.4")
    bad = curve("21_histogram_bin_0-100", "minimax-m2")
    axA.plot(THREADS, THREADS, "--", color=MUTE, lw=1.3, label="ideal (linear)", zorder=1)
    axA.axhline(1, color=C_RACE, lw=1.1, ls=(0, (1, 2)), zorder=1)
    if good:
        axA.plot(THREADS, good, "-o", color=C_L2, lw=2.8, ms=9, zorder=4,
                 markeredgecolor="white", markeredgewidth=1.2,
                 label="GPT-5.4  (reduction): 6.5$\\times$")
    if bad:
        axA.plot(THREADS, bad, "-D", color=C_RACE, lw=2.8, ms=9, zorder=5,
                 markeredgecolor="white", markeredgewidth=1.2,
                 label="MiniMax-M2 (atomic): 0.33$\\times$")
    axA.text(8, 1.15, "slower than\n1 thread", color=C_RACE, fontsize=8.4,
             ha="right", va="bottom", style="italic")
    despine(axA)
    axA.set_xscale("log", base=2); axA.set_yscale("log", base=2)
    axA.set_xticks(THREADS); axA.set_xticklabels(THREADS)
    axA.set_yticks([0.5, 1, 2, 4, 8]); axA.set_yticklabels(["0.5", "1", "2", "4", "8"])
    axA.set_xlabel("threads"); axA.set_ylabel("self-speedup  T(1)/T(p)")
    axA.set_title("(a)  Same task, same \u201crobust\u201d verdict\u2014opposite scaling", fontsize=10.5)
    axA.legend(fontsize=8.6, loc="upper left")

    # ---- Panel B: speedup(8) for every accepted program, sorted, by idiom ----
    idcol = {"reduction": C_L2, "plain_for": "#56B4E9", "critical": C_L1,
             "atomic": C_RACE, "single/master": "#CC79A7", "no_omp": "#000000", "?": MUTE}
    pts = [(r["self_speedup8"], osync.get((r["task"], r["model"]), "?"))
           for r in scal if r.get("self_speedup8")]
    pts.sort(key=lambda z: z[0])
    xs = np.arange(len(pts))
    ys = [p[0] for p in pts]
    cols = [idcol.get(p[1], MUTE) for p in pts]
    axB.axhline(8, color=MUTE, ls="--", lw=1.2, zorder=1)
    axB.text(len(pts) - 0.5, 8.05, "ideal 8$\\times$", color=MUTE, fontsize=8.4,
             ha="right", va="bottom")
    axB.axhline(1, color=C_RACE, ls=(0, (1, 2)), lw=1.2, zorder=1)
    axB.text(len(pts) - 0.5, 1.05, "break-even", color=C_RACE, fontsize=8.4,
             ha="right", va="bottom", style="italic")
    axB.bar(xs, ys, color=cols, edgecolor="white", lw=0.4, width=0.9, zorder=3)
    despine(axB)
    axB.set_xticks([])
    axB.set_xlabel(f"{len(pts)} programs the verifier accepted as robust (sorted)")
    axB.set_ylabel("self-speedup on 8 threads")
    axB.set_ylim(0, 8.6)
    axB.set_title("(b)  Robust correctness spans 0.33$\\times$\u20137.9$\\times$ scaling", fontsize=10.5)
    from matplotlib.patches import Patch
    order = ["reduction", "plain_for", "critical", "atomic"]
    leg = [Patch(facecolor=idcol[k], edgecolor="white", label=k) for k in order]
    axB.legend(handles=leg, fontsize=8.6, loc="upper left", ncol=1, title="idiom",
               title_fontsize=8.6)

    fig.suptitle("Configuration-robust correctness is necessary but not sufficient: "
                 "a serial program scores 100%",
                 fontsize=12, weight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.28, top=0.84, bottom=0.14)
    fig.savefig(FIG / "fig_scaling.pdf")
    fig.savefig(FIG / "fig_scaling.png", dpi=150)
    plt.close(fig)
    print("wrote fig_scaling")


if __name__ == "__main__":
    main()
