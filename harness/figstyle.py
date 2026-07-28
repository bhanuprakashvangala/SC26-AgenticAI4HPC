"""State-of-the-art publication style for the figure suite.

One place to control palette, typography, and helpers so every figure has a
consistent, refined, journal-quality look. Design principles: Okabe-Ito
colourblind-safe palette, Arial, no chartjunk (top/right spines removed, minimal
horizontal grid), direct labelling over legends where possible, generous
whitespace, and a clear visual hierarchy that guides the eye to the takeaway.
"""
from __future__ import annotations
import math


# ---- Okabe-Ito colourblind-safe palette (the scientific-publishing standard) ----
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_BLUE = "#0072B2"
OI_VERM = "#D55E00"
OI_PURPLE = "#CC79A7"

# semantic roles (kept stable across every figure)
C_L0 = "#9AA0A6"       # single-shot baseline -- neutral slate
C_L1 = OI_ORANGE       # execute-once -- amber
C_L2 = OI_GREEN        # differential -- bluish green
C_RACE = OI_VERM       # silent races / hazards -- vermillion
C_FRONTIER = OI_BLUE
C_OPEN = OI_VERM
INK = "#1A1A1A"
MUTE = "#6B7280"
GRIDC = "#E6E6E6"

CCOL = {"single_shot": C_L0, "execute_once": C_L1, "differential_verify": C_L2}
CLABEL = {"single_shot": "L0 single-shot",
          "execute_once": "L1 execute-once",
          "differential_verify": "L2 differential"}


def apply_style(plt):
    import matplotlib as mpl
    base = "DejaVu Sans"
    for fam in ("Arial", "Helvetica", "Segoe UI"):
        try:
            if mpl.font_manager.findfont(fam, fallback_to_default=False):
                base = fam
                break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family": base,
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelsize": 11.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#B7BCC2",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDC,
        "grid.linewidth": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "ytick.major.width": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.2,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "svg.fonttype": "none",
    })


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def despine(ax, keep=("left", "bottom"), grid_axis="y"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0, which="both", axis="x")
    ax.grid(axis=grid_axis)
    ax.grid(axis="x", visible=False)


def bar_value_labels(ax, bars, fmt="{:.0f}", dy=1.6, fontsize=10, color=INK,
                     weight="semibold"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color,
                weight=weight)


def lighten(hex_color, amount=0.5):
    """Blend a hex colour toward white by `amount` (0=same, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"
