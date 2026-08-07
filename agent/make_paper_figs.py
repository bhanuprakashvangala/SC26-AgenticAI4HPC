"""
make_paper_figs.py  --  the two figures the reframed paper needs, drawn from the frozen data.

  figs/projection.(pdf|png)  the degeneracy, measured: each accepted program's reward BEFORE vs
                             AFTER deleting every #pragma omp. Under R_corr every point stays on the
                             diagonal (no penalty for going serial); under R_gate every point
                             collapses to the 1/n serial floor.
  figs/scaling.(pdf|png)     sorted 8-thread self-speedup for all accepted programs, coloured by the
                             synchronization idiom the model chose, with the below-serial band shaded
                             -- all at the identical correctness-only reward of 1.0.

Offline. `python -m agent.make_paper_figs`.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_core as DC  # noqa: E402

NMAX = 8
OUT = os.path.join(DC._ROOT, "paper_agentic", "figs")
os.makedirs(OUT, exist_ok=True)

# a restrained, print-friendly palette
ACCENT = "#0F6CBD"
MUTED = "#8A8886"
IDIOM_COLOR = {"reduction": "#107C10", "plain_for": "#0F6CBD",
               "critical": "#D13438", "atomic": "#CA5010", None: MUTED, "unknown": MUTED}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#444",
        "axes.linewidth": 0.7, "axes.grid": True, "grid.color": "#E1DFDD",
        "grid.linewidth": 0.6, "figure.dpi": 200, "savefig.bbox": "tight",
    })


def fig_projection(sess):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    floor = 1.0 / NMAX
    for p in sess.programs:
        proj = p.get("serial_projection") or {}
        stays = proj.get("predicted_still_correct")
        # LEFT: correctness-only reward, before(x) vs after(y)
        bx = p["reward_correctness_only"]
        by = bx if stays else 0.0
        axL.scatter(bx, by, s=16, color=(ACCENT if stays else "#D13438"),
                    alpha=0.7, edgecolor="white", linewidth=0.4, zorder=3)
        # RIGHT: gated reward, before(x)=real gated reward, after(y)=serial floor
        gx = p["reward_efficiency_gated"] or 0.0
        gy = floor if stays else 0.0
        axR.scatter(gx, gy, s=16, color=ACCENT, alpha=0.7,
                    edgecolor="white", linewidth=0.4, zorder=3)

    for ax in (axL, axR):
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=0.9, zorder=1)
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("reward before stripping")
    axL.set_ylabel("reward after stripping")
    axL.set_title(r"$R_{\mathrm{corr}}$: correctness only", fontsize=8.5)
    axR.set_title(r"$R_{\mathrm{gate}}$: efficiency-gated", fontsize=8.5)
    axR.axhline(floor, color="#107C10", linewidth=0.9, linestyle=":", zorder=1)
    axR.text(0.98, floor + 0.03, r"$1/n$ serial floor", ha="right", va="bottom",
             fontsize=7, color="#107C10")
    axL.text(0.5, 0.42, "every program\nstays on the diagonal\n(no penalty to go serial)",
             ha="center", va="center", fontsize=6.8, color=MUTED)
    fig.suptitle("Deleting every OpenMP directive: what each reward charges for it",
                 fontsize=9, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "projection.%s" % ext))
    plt.close(fig)


def fig_scaling(sess):
    progs = [p for p in sess.programs if p["self_speedup8"] is not None]
    progs = sorted(progs, key=lambda p: p["self_speedup8"])
    xs = list(range(len(progs)))
    ys = [p["self_speedup8"] for p in progs]
    cols = [IDIOM_COLOR.get((p["sync_idiom"] or {}).get("primary"), MUTED) for p in progs]

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axhspan(0, 1.0, color="#FDE7E9", zorder=0)  # below-serial band
    ax.axhline(1.0, color="#D13438", linewidth=0.9, linestyle="--", zorder=2)
    ax.bar(xs, ys, color=cols, width=0.9, zorder=3, edgecolor="white", linewidth=0.2)
    ax.set_xlim(-1, len(progs))
    ax.set_ylabel("self-speedup at 8 threads")
    ax.set_xlabel("accepted programs, sorted (all correct; all reward $R_{\\mathrm{corr}}{=}1$)")
    ax.text(1, 0.9, "slower than serial", fontsize=7, color="#A4262C", va="top")
    ax.text(len(progs) - 1, max(ys) * 0.96,
            "range %.2f--%.2f, median %.2f" % (ys[0], ys[-1],
                                               sorted(ys)[len(ys) // 2]),
            ha="right", va="top", fontsize=7.5, color="#333")
    handles = [plt.Rectangle((0, 0), 1, 1, color=IDIOM_COLOR[k])
               for k in ("reduction", "plain_for", "critical", "atomic")]
    ax.legend(handles, ["reduction", "plain for", "critical", "atomic"],
              title="sync idiom", fontsize=7, title_fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("What the correctness-only reward cannot see: 20x scaling spread at identical reward",
                 fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "scaling.%s" % ext))
    plt.close(fig)


def fig_deadgroups(sess):
    """E3: predicted fraction of zero-variance (dead) GRPO groups vs k, from measured per-task
    pass rates -- under R_corr (saturated correctness kills the gradient) vs R_gate (continuous)."""
    import json
    runs = [json.loads(l) for l in open(os.path.join(DC._RESULTS, "pareval_runs.jsonl"),
                                        encoding="utf-8") if l.strip()]
    ss = [r for r in runs if r["condition"] == "single_shot"]
    from collections import defaultdict
    bt = defaultdict(list)
    for r in ss:
        bt[r["task"]].append(str(r.get("robust")).lower() == "true" or r.get("verdict") == "ROBUST_CORRECT")
    p_t = [sum(v) / len(v) for v in bt.values() if v]
    ks = list(range(2, 17))
    corr = [100 * sum(p ** k + (1 - p) ** k for p in p_t) / len(p_t) for k in ks]
    # under the gate (1[c]*continuous f), a group dies only if ALL k are incorrect: (1-p)^k
    gate = [100 * sum((1 - p) ** k for p in p_t) / len(p_t) for k in ks]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot(ks, corr, "-o", color="#D13438", markersize=3, label=r"$R_{\mathrm{corr}}$ (correctness only)")
    ax.plot(ks, gate, "-s", color="#107C10", markersize=3, label=r"$R_{\mathrm{gate}}$ (efficiency-gated)")
    ax.set_xlabel("group size $k$")
    ax.set_ylabel("% zero-variance groups (no gradient)")
    ax.set_ylim(-3, 100)
    ax.set_title("Predicted GRPO signal collapse", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="center right")
    ax.text(8.2, corr[6] - 8, "%.0f%% dead\nat $k{=}8$" % corr[6], fontsize=7, color="#A4262C")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "deadgroups.%s" % ext))
    plt.close(fig)


def fig_hackheat():
    """Reward x hack heatmap: what each reward pays a CORRECT adversarial candidate (0..1)."""
    import numpy as np
    import parallel_gate as PG
    rids = list(PG.REWARDS.keys())
    hacks = [h["id"] for h in PG.HACKS]
    M = np.array([[PG.REWARDS[r][1](True, h["effect"]["sigma"], h["effect"]["sref"],
                                    h["effect"]["timing_ok"]) for h in PG.HACKS] for r in rids])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hacks))); ax.set_xticklabels(hacks, fontsize=8)
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels(["%s %s" % (r, PG.REWARDS[r][0]) for r in rids], fontsize=7.5)
    for i in range(len(rids)):
        for j in range(len(hacks)):
            v = M[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.6,
                    color="white" if (v > 0.72 or v < 0.14) else "#222")
    # frame OUR row
    oi = rids.index(PG.OURS)
    ax.add_patch(plt.Rectangle((-0.5, oi - 0.5), len(hacks), 1, fill=False,
                               edgecolor=ACCENT, linewidth=2))
    ax.set_title("Reward the correct adversarial candidate earns\n(1.00 = fully fooled; ours boxed)",
                 fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "hackheat.%s" % ext))
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec, fs=7.6, tc="#1a1a1a"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((xy[0], xy[1]), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.1, facecolor=fc, edgecolor=ec))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, wrap=True)


def _arrow(ax, a, b, color="#555"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=color, shrinkA=2, shrinkB=2))


def fig_architecture():
    """The grounded agentic diagnostician: task -> tools(pool) -> agent -> critic -> verdict."""
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    LB, GR, RD, PL = "#EFF6FC", "#E7F5EC", "#FBE9EA", "#F3F0FA"
    y, h = 24, 12
    # five main stages, evenly spaced with generous width
    _box(ax, (1, y), 17, h, "generated\nOpenMP program", LB, ACCENT, fs=8.0)
    _box(ax, (21, y), 17, h, "measurement\ntools", "#FFFFFF", "#8A8886", fs=8.0)
    _box(ax, (41, y), 17, h, "diagnostician\nagent", LB, ACCENT, fs=8.0)
    _box(ax, (61, y), 18, h, "performance-\ncritic", GR, "#0E7A3B", fs=8.0)
    _box(ax, (82, y), 17, h, "grounded\nverdict + audit", RD, "#B10E1C", fs=8.0)
    for x0, x1 in [(18, 21), (38, 41), (58, 61), (79, 82)]:
        _arrow(ax, (x0, y + h / 2), (x1, y + h / 2))
    # tool detail caption under the tools box
    ax.text(29.5, y - 2.2, "compile · thread-sweep timing\nrace check (Archer) · serial projection",
            ha="center", va="top", fontsize=6.4, color="#616161")
    # frozen pool feeds the tools
    _box(ax, (20, 3), 19, 8, "frozen run pool\n63 programs · real timings", PL, "#8661C5", fs=6.8)
    _arrow(ax, (29.5, 11), (29.5, 15.5), color="#8661C5")
    # critic detail + fail loop back to the agent
    ax.text(70, y - 2.2, "every claim $\\subseteq$ measurements?", ha="center", va="top",
            fontsize=6.6, color="#0E7A3B")
    ax.add_patch(FancyArrowPatch((66, y + h), (49, y + h), connectionstyle="arc3,rad=-0.4",
                 arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#B10E1C",
                 linestyle=(0, (3, 2))))
    ax.text(57.5, y + h + 6.0, "FAIL: unmeasured claim $\\rightarrow$ revise",
            fontsize=6.8, color="#B10E1C", ha="center")
    ax.text(50, 0.2, "nothing is asserted that a measurement does not return",
            fontsize=7.4, color="#444", ha="center", style="italic")
    fig.savefig(os.path.join(OUT, "architecture.pdf")); fig.savefig(os.path.join(OUT, "architecture.png"))
    plt.close(fig)


if __name__ == "__main__":
    _style()
    sess = DC.Session(verbose=False)
    fig_projection(sess)
    fig_scaling(sess)
    fig_deadgroups(sess)
    fig_hackheat()
    fig_architecture()
    print("wrote projection, scaling, deadgroups, hackheat, architecture {pdf,png} to",
          os.path.relpath(OUT, DC._ROOT))
