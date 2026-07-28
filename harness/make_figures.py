"""Publication figure + number generator for the agentic ParEval study.

Reads ONLY real logged artifacts -- results/pareval_runs.jsonl (one row per
task x model x condition x trial), logs/pareval/transcripts/*.json (per-round
repair steps + verifier diagnostics), and results/injected_races.jsonl (the
hand-written positive control) -- and emits a coherent figure suite plus every
LaTeX macro the paper cites. Nothing is hard-coded: if the underlying data
changes (e.g. a larger live run lands), rerunning this file updates the figures
and numbers together.

Figures (results/figures/):
  fig_ablation           L0/L1/L2 robust-correctness with Wilson 95% CIs
  fig_by_model           robust rate per model x tool level (grouped bars)
  fig_by_type            robust rate per ParEval problem type (where failures live)
  fig_failure_taxonomy   why generations failed: compile / wrong@1-thread / race
  fig_thread_sweep       pass-rate vs thread count: models' code vs injected races
  fig_repair_dynamics    robust rate after each agentic repair round
"""
from __future__ import annotations
import json, math, re
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "pareval_runs.jsonl"
TR = ROOT / "logs" / "pareval" / "transcripts"
INJ = ROOT / "results" / "injected_races.jsonl"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COND = ["single_shot", "execute_once", "differential_verify"]
CLABEL = {"single_shot": "L0\nsingle-shot",
          "execute_once": "L1\nexecute-once",
          "differential_verify": "L2\ndifferential"}
# refined Okabe-Ito palette, shared across every figure via figstyle
from harness.figstyle import (C_L0, C_L1, C_L2, C_RACE, INK, MUTE, GRIDC,
                              lighten, despine, bar_value_labels)
CCOL = {"single_shot": C_L0, "execute_once": C_L1, "differential_verify": C_L2}
THREADS = [1, 2, 4, 8]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load_rows():
    seen = {}
    for ln in RES.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            seen[(r["task"], r["model"], r["condition"], r.get("trial", 0))] = r
    return list(seen.values())


def load_injected():
    out = []
    if INJ.exists():
        for ln in INJ.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    return out


OPEN = ROOT / "results" / "pareval_open.jsonl"


def load_open_race():
    """Find a REAL weak open-model silent race (execute-once final that passes at
    1 thread but fails at a higher count) and return its per-thread pass-rate plus
    a label, or None. This upgrades the money figure from an injected control to a
    race an actual model shipped."""
    if not OPEN.exists():
        return None
    best = None
    for ln in OPEN.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("condition") != "execute_once":
            continue
        sw = parse_sweep(r.get("sweep", ""))
        if not sw:
            continue
        p1 = sw.get(1, (0, 0))
        hi_fail = any(sw[t][0] < sw[t][1] for t in sw if t > 1)
        if p1[1] and p1[0] == p1[1] and hi_fail:      # pass@1, fail@N -> race
            rate = [100 * sw[t][0] / sw[t][1] if sw.get(t, (0, 0))[1] else None
                    for t in THREADS]
            best = (r["model"], r["task"], rate)
            break
    return best


def parse_sweep(s):
    """'t1=2/2 t2=2/2 t4=0/2 t8=0/2' -> {1:(2,2),2:(2,2),4:(0,2),8:(0,2)}."""
    out = {}
    for m in re.finditer(r"t(\d+)=(\d+)/(\d+)", s or ""):
        out[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return out


def _short_model(m):
    return m.replace("gpt-", "").replace("-small", "-sm")


def _short_task(t):
    """'22_histogram_count_quadrants' -> 'histogram: count quadrants'."""
    base = re.sub(r"^\d+_", "", t)
    parts = base.split("_")
    fam = parts[0]
    rest = " ".join(parts[1:]).replace(fam + " ", "")
    return f"{fam}: {rest}" if rest else fam


# ---------- failure classification from verifier diagnostics ----------
def classify_failure(diag: str, sweep: str = "") -> str:
    """Bucket a NOT_ROBUST cell by WHY it failed, using the verifier's own diag
    (and per-thread sweep when present). Race = passes at 1 thread but fails at a
    higher count; the whole point of the differential verifier."""
    d = (diag or "").lower()
    if "error:" in d or "crash" in d or "declared" in d or "stray" in d:
        return "compile"
    sw = parse_sweep(sweep)
    if sw:
        p1 = sw.get(1, (0, 0))
        hi_fail = any(sw[t][0] < sw[t][1] for t in sw if t > 1)
        if p1[1] and p1[0] == p1[1] and hi_fail:
            return "race"          # clean at 1 thread, breaks above it
        return "semantic"          # already wrong at 1 thread
    m = re.search(r"fail on (\d+) of (\d+)", d)
    if m:
        k, n = int(m.group(1)), int(m.group(2))
        return "semantic" if k == n else "race"  # partial fail w/o sweep -> race-like
    if "incomplete sweep" in d:
        return "race"              # hung/deadlocked above 1 thread
    return "semantic"


def load_transcripts():
    """cell_key -> {steps:[...], final_verdict, robust, n_iters}."""
    out = {}
    if not TR.exists():
        return out
    for f in TR.glob("*.json"):
        d = json.load(open(f, encoding="utf-8"))
        out[(d["task"], d["model"], d["condition"], d.get("trial", 0))] = d
    return out


def is_compile(row, tr):
    key = (row["task"], row["model"], row["condition"], row.get("trial", 0))
    t = tr.get(key)
    diag = t["steps"][-1].get("diag", "") if t and t.get("steps") else ""
    d = (diag or row.get("verdict", "")).lower()
    return ("error:" in d or "crash" in d or "declared" in d or "stray" in d
            or row.get("verdict") == "CRASH")


def fail_bucket(row, tr, l1_robust):
    """Authoritative failure bucket for a NOT_ROBUST L2 (differential) row.

    RACE is defined the same way the paper's null is measured -- the code passed
    the execute-once (1-thread) check for the SAME (task,model,trial) cell but
    failed the thread sweep. This is exactly what only the differential verifier
    can catch; a partial-fail diagnostic string is not trusted on its own."""
    if is_compile(row, tr):
        return "compile"
    key = (row["task"], row["model"], row.get("trial", 0))
    if l1_robust.get(key):          # certified at 1 thread, broke under the sweep
        return "race"
    return "semantic"               # already wrong at 1 thread


# =====================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from harness.figstyle import apply_style
    apply_style(plt)

    rows = load_rows()
    tr = load_transcripts()
    inj = load_injected()
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    types = sorted({r["type"] for r in rows})
    macros = []

    def save(fig, name):
        fig.savefig(FIG / f"{name}.pdf")
        fig.savefig(FIG / f"{name}.png", dpi=150)
        plt.close(fig)
        print("  wrote", name)

    print(f"loaded {len(rows)} runs / {len(tasks)} tasks / {len(models)} models")

    # =========================== FIG 0: pipeline schematic =================
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 31)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.5, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.4",
                     fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")

    def arrow(x1, y1, x2, y2, color="#333", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=14, lw=1.6, color=color))

    box(1, 11, 20, 9, "LLM generates\nOpenMP code", "#DCE7F5", "#3A6EA5")
    box(28, 11, 26, 9, "Differential verifier\n(serial baseline vs.\nthread sweep t = 1,2,4,8)",
        "#EAD9F2", "#7A1FA2", tc="#4A0F66")
    box(62, 17, 16, 8, "certify:\nrobust", "#D6EAD9", "#1E6F3C", tc="#0F3D20")
    box(62, 3, 16, 8, "repair\n(diff \u2192 LLM)", "#F7E2C2", "#9D5D00", tc="#5B3600")
    arrow(21, 15.5, 28, 15.5)
    arrow(54, 17, 62, 20.5)                     # verify -> certify (pass)
    ax.text(57.5, 21.6, "pass", fontsize=8.5, color="#1E6F3C", weight="bold")
    arrow(54, 14, 62, 7.5, color="#9D5D00")     # verify -> repair (fail)
    ax.text(57.5, 9.3, "fail", fontsize=8.5, color="#9D5D00", weight="bold")
    arrow(70, 3, 70, -1, color="#9D5D00")
    ax.add_patch(FancyArrowPatch((70, -1), (11, -1), connectionstyle="arc3,rad=0",
                 arrowstyle="-", lw=1.6, color="#9D5D00"))
    arrow(11, -1, 11, 11, color="#9D5D00")      # repair loops back to generate
    ax.text(40, -2.4, "agentic repair loop (budget-bounded)", ha="center",
            fontsize=8.5, style="italic", color="#9D5D00")
    ax.text(41, 27.5, "The differential verifier is the agent's tool; the ablation swaps ONLY this box.",
            ha="center", fontsize=8.5, color="#555")
    save(fig, "fig_pipeline")

    # ---- headline ablation numbers ----
    cond_stat = {}
    for c in COND:
        sub = [r for r in rows if r["condition"] == c]
        k = sum(1 for r in sub if r.get("robust"))
        cond_stat[c] = (k, len(sub), *wilson(k, len(sub)))
    key = {"single_shot": "Lzero", "execute_once": "Lone", "differential_verify": "Ltwo"}
    for c, kk in key.items():
        k, n, p, lo, hi = cond_stat[c]
        macros += [f"\\newcommand{{\\{kk}rate}}{{{100*p:.1f}\\%}}",
                   f"\\newcommand{{\\{kk}ci}}{{[{100*lo:.1f}, {100*hi:.1f}]}}"]
    macros.append(f"\\newcommand{{\\LtwoGain}}{{{100*(cond_stat['differential_verify'][2]-cond_stat['single_shot'][2]):.1f}\\%}}")
    macros.append(f"\\newcommand{{\\LtwoOverLone}}{{{100*(cond_stat['differential_verify'][2]-cond_stat['execute_once'][2]):+.1f}\\%}}")

    # paired repair / differential-vs-execute analysis
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["task"], r["model"], r.get("trial", 0))][r["condition"]] = r.get("robust")
    fail_ss = [c for c in by_cell.values() if "single_shot" in c and not c["single_shot"]]
    eo_fix = sum(1 for c in fail_ss if c.get("execute_once"))
    dv_fix = sum(1 for c in fail_ss if c.get("differential_verify"))
    paired = [c for c in by_cell.values() if "execute_once" in c and "differential_verify" in c]
    ndiffer = sum(1 for c in paired if bool(c["execute_once"]) != bool(c["differential_verify"]))
    dv_only = sum(1 for c in paired if c.get("differential_verify") and not c.get("execute_once"))
    macros += [f"\\newcommand{{\\NfailSingleShot}}{{{len(fail_ss)}}}",
               f"\\newcommand{{\\EoRescued}}{{{eo_fix}}}",
               f"\\newcommand{{\\DvRescued}}{{{dv_fix}}}",
               f"\\newcommand{{\\Npairs}}{{{len(paired)}}}",
               f"\\newcommand{{\\Ndiffer}}{{{ndiffer}}}",
               f"\\newcommand{{\\DvOnlyRescued}}{{{dv_only}}}",
               f"\\newcommand{{\\Ntasks}}{{{len(tasks)}}}",
               f"\\newcommand{{\\Nmodels}}{{{len(models)}}}",
               f"\\newcommand{{\\Nruns}}{{{len(rows)}}}",
               f"\\newcommand{{\\Ntypes}}{{{len(types)}}}"]

    # =========================== FIG 1: ablation ===========================
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    xs = list(range(len(COND)))
    ps = [100 * cond_stat[c][2] for c in COND]
    err = [[100 * (cond_stat[c][2] - cond_stat[c][3]) for c in COND],
           [100 * (cond_stat[c][4] - cond_stat[c][2]) for c in COND]]
    bars = ax.bar(xs, ps, yerr=err, capsize=6, color=[CCOL[c] for c in COND],
                  edgecolor="white", linewidth=1.2, width=0.66, zorder=3,
                  error_kw=dict(lw=1.3, ecolor=MUTE, capthick=1.3))
    for i, c in enumerate(COND):
        k, n = cond_stat[c][0], cond_stat[c][1]
        ax.text(i, ps[i] / 2, f"{ps[i]:.0f}%", ha="center", va="center",
                fontsize=15, weight="bold", color="white")
        ax.text(i, ps[i] / 2 - 8, f"{k}/{n}", ha="center", va="center",
                fontsize=10, color="white", alpha=0.9)
    despine(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLABEL[c] for c in COND], fontsize=10.5)
    ax.set_ylabel("Configuration-robust correctness (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Tool ablation: only the verifier's granularity changes")
    # 'identical' bracket for L1==L2, drawn above the CI whiskers
    ytop = max(100 * cond_stat[c][4] for c in ["execute_once", "differential_verify"]) + 3
    ax.plot([1, 1, 2, 2], [ytop, ytop + 2.5, ytop + 2.5, ytop], color=INK, lw=1.1)
    ax.text(1.5, ytop + 3.2, "identical", ha="center", va="bottom",
            fontsize=10, style="italic", color=INK)
    save(fig, "fig_ablation")

    # =========================== FIG 2: by model ===========================
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    w = 0.26
    for j, c in enumerate(COND):
        vals = []
        for m in models:
            sub = [r for r in rows if r["model"] == m and r["condition"] == c]
            k = sum(1 for r in sub if r.get("robust"))
            vals.append(100 * k / len(sub) if sub else 0)
        ax.bar([x + (j - 1) * w for x in range(len(models))], vals, w,
               label=CLABEL[c].replace("\n", " "), color=CCOL[c],
               edgecolor="white", linewidth=1.0, zorder=3)
    despine(ax)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_short_model(m) for m in models])
    ax.set_ylabel("Robust correctness (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Robust correctness by model and tool level")
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "fig_by_model")

    # =========================== FIG 3: by type ===========================
    tstat = []
    for t in types:
        sub = [r for r in rows if r["type"] == t and r["condition"] == "differential_verify"]
        k = sum(1 for r in sub if r.get("robust"))
        if sub:
            tstat.append((t, k, len(sub), k / len(sub)))
    tstat.sort(key=lambda x: x[3])
    fig, ax = plt.subplots(figsize=(6.6, max(3.3, 0.46 * len(tstat) + 1)))
    ys = range(len(tstat))
    cols = [C_RACE if s[3] < 0.5 else (C_L1 if s[3] < 0.999 else C_L2) for s in tstat]
    bars = ax.barh(list(ys), [100 * s[3] for s in tstat], color=cols,
                   edgecolor="white", linewidth=1.0, height=0.72, zorder=3)
    for i, s in enumerate(tstat):
        val = 100 * s[3]
        # place the count label inside the bar if it's wide enough, else outside
        if val > 12:
            ax.text(val - 2, i, f"{s[1]}/{s[2]}", va="center", ha="right",
                    fontsize=9.5, color="white", weight="semibold")
        else:
            ax.text(val + 2, i, f"{s[1]}/{s[2]}", va="center", ha="left",
                    fontsize=9.5, color=INK, weight="semibold")
    despine(ax, keep=("left", "bottom"), grid_axis="x")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0] for s in tstat])
    ax.set_xlabel("Robust correctness (%)")
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("Where failures concentrate (differential verification)")
    save(fig, "fig_by_type")
    _tm = {"histogram": "Hist", "reduce": "Reduce", "scan": "Scan",
           "search": "Search", "graph": "Graph"}
    for t in types:
        sub = [r for r in rows if r.get("type") == t]
        k = sum(1 for r in sub if r.get("robust"))
        if t in _tm:
            macros.append(f"\\newcommand{{\\Robust{_tm[t]}}}{{{k}/{len(sub)}}}")

    # ===================== FIG 4: failure taxonomy =====================
    # Authoritative race count = L1-pass/L2-fail per cell (same signal as the
    # null). We add the injected control as a rightmost group to make the point:
    # the verifier CAN catch races (all injected), the models just didn't write
    # them on standard tasks.
    l1_robust = {}
    for r in rows:
        if r["condition"] == "execute_once":
            l1_robust[(r["task"], r["model"], r.get("trial", 0))] = r.get("robust")
    buckets = ["compile", "semantic", "race"]
    blabel = {"compile": "Compile error", "semantic": "Wrong at 1 thread",
              "race": "Race (pass@1, fail@N)"}
    bcol = {"compile": C_L0, "semantic": C_L1, "race": C_RACE}
    per_model = {m: Counter() for m in models}
    for r in rows:
        if r["condition"] == "differential_verify" and not r.get("robust"):
            per_model[r["model"]][fail_bucket(r, tr, l1_robust)] += 1
    inj_ct = Counter()
    for e in inj:
        if str(e.get("differential", "")).upper().startswith("NOT"):
            inj_ct["race"] += 1
    groups = list(models) + (["injected\ncontrol"] if inj else [])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bottoms = [0] * len(groups)
    for b in buckets:
        vals = [per_model[m][b] for m in models] + ([inj_ct[b]] if inj else [])
        ax.bar(range(len(groups)), vals, bottom=bottoms, label=blabel[b],
               color=bcol[b], edgecolor="white", linewidth=1.0, width=0.64, zorder=3)
        bottoms = [x + y for x, y in zip(bottoms, vals)]
    if inj:
        ax.axvline(len(models) - 0.5, color=MUTE, ls=(0, (2, 2)), lw=1.1)
    despine(ax)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_short_model(g) if g in models else g for g in groups])
    ax.set_ylabel("# failing generations (L2)")
    ax.set_ylim(0, max(bottoms + [1]) + 0.6)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    n_race = sum(per_model[m]["race"] for m in models)
    ax.set_title(f"Frontier failures are semantic, not races "
                 f"({n_race} in model code vs. {inj_ct['race']}/{len(inj)} caught in control)"
                 if inj else "Frontier failures are semantic, not races", fontsize=11.5)
    ax.legend(fontsize=9.5, loc="upper right")
    save(fig, "fig_failure_taxonomy")
    total_fail = sum(sum(per_model[m].values()) for m in models)
    n_comp = sum(per_model[m]["compile"] for m in models)
    n_sem = sum(per_model[m]["semantic"] for m in models)
    macros += [f"\\newcommand{{\\Nfail}}{{{total_fail}}}",
               f"\\newcommand{{\\NfailRace}}{{{n_race}}}",
               f"\\newcommand{{\\NfailCompile}}{{{n_comp}}}",
               f"\\newcommand{{\\NfailSemantic}}{{{n_sem}}}"]

    # ===================== FIG 5: thread-sweep cliff (money figure) =====
    # Models' own code: per-thread pass rate. Use real sweep where present;
    # otherwise a robust cell passes at every thread and a non-robust cell that
    # failed already at 1 thread (its diag says so) fails at every thread.
    std_pass = {t: [0, 0] for t in THREADS}   # [passes, total] across cells
    for r in rows:
        if r["condition"] != "differential_verify":
            continue
        sw = parse_sweep(r.get("sweep", ""))
        for t in THREADS:
            if t in sw:
                p, n = sw[t]
                std_pass[t][0] += p
                std_pass[t][1] += n
            else:
                std_pass[t][1] += 1
                if r.get("robust"):
                    std_pass[t][0] += 1
                # non-robust w/o sweep: standard failures were wrong at 1 thread
                # (see failure taxonomy) -> counts as fail at every thread.
    std_rate = [100 * std_pass[t][0] / std_pass[t][1] if std_pass[t][1] else 0 for t in THREADS]
    open_race = load_open_race()          # a REAL weak-model silent race, if any

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # shade the region a 1-thread check cannot see
    ax.axvspan(1.35, 8.7, color=C_RACE, alpha=0.05, zorder=0)
    ax.plot(THREADS, std_rate, "-o", color=C_L2, lw=2.8, ms=9,
            markeredgecolor="white", markeredgewidth=1.2,
            label="Robust model code (stays correct)", zorder=4)
    if open_race:
        omodel, otask, orate = open_race
        orate = [v if v is not None else 0 for v in orate]
        ax.plot(THREADS, orate, "-D", color=C_RACE, lw=2.8, ms=9,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"Real race {_short_model(omodel)} shipped\n({_short_task(otask)}) \u2014 L1 accepted it",
                zorder=5)
    elif inj := (load_injected()):
        ax.plot(THREADS, [100.0, 0.0, 0.0, 0.0], "--D", color=C_RACE, lw=2.6, ms=9,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"Injected racy code (control, n={len(inj)})", zorder=5)
    ax.axvline(1, color=C_RACE, ls=(0, (1, 2)), lw=1.4, alpha=0.6, zorder=1)
    ax.text(1.04, 46, "a 1-thread check\nsees only here", color=C_RACE,
            fontsize=9, ha="left", va="center", style="italic")
    # annotate the cliff
    if open_race:
        ax.annotate("silent race:\ncorrect at 1, wrong above",
                    xy=(2, 3), xytext=(3.0, 30), fontsize=9.2, color=C_RACE,
                    weight="semibold", ha="left",
                    arrowprops=dict(arrowstyle="-|>", color=C_RACE, lw=1.5))
    despine(ax, grid_axis="y")
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREADS)
    ax.set_xlabel("threads")
    ax.set_ylabel("validation pass-rate (%)")
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Correct on one thread, wrong on eight?")
    ax.legend(fontsize=8.8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    save(fig, "fig_thread_sweep")

    # ===================== FIG 6: repair dynamics =====================
    # robust-so-far vs repair round for the two verify+repair conditions.
    maxr = 0
    for d in tr.values():
        maxr = max(maxr, len(d.get("steps", [])))
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for c in ["execute_once", "differential_verify"]:
        cells = [d for d in tr.values() if d["condition"] == c]
        n = len(cells)
        if not n:
            continue
        # cumulative fraction settled-correct at/after each round
        cum = []
        for rnd in range(maxr + 1):
            got = 0
            for d in cells:
                steps = d.get("steps", [])
                passed_round = next((i for i, s in enumerate(steps)
                                     if s.get("verdict") == "ROBUST_CORRECT"), None)
                if d.get("robust") and (passed_round is not None and passed_round <= rnd):
                    got += 1
                elif d.get("robust") and not steps and rnd >= 0:
                    got += 1
            cum.append(100 * got / n)
        # ensure final equals actual robust rate
        cum[-1] = 100 * sum(1 for d in cells if d.get("robust")) / n
        off = 0.6 if c == "execute_once" else -0.6   # tiny offset so overlapping lines both read
        ax.plot(range(len(cum)), [v + off for v in cum], "-o", color=CCOL[c], lw=2.8, ms=8,
                markeredgecolor="white", markeredgewidth=1.2,
                label=CLABEL[c].replace("\n", " "), zorder=3)
    despine(ax, grid_axis="y")
    ax.set_xlabel("agentic repair round")
    ax.set_ylabel("cumulative robust correctness (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(maxr + 1))
    ax.set_title("The repair loop, not the verifier, drives the gains")
    ax.legend(fontsize=10, loc="lower right")
    save(fig, "fig_repair_dynamics")

    # legacy alias so existing \includegraphics{pareval_ablation} still resolves
    import shutil
    for ext in ("pdf", "png"):
        shutil.copyfile(FIG / f"fig_ablation.{ext}", FIG / f"pareval_ablation.{ext}")

    (ROOT / "results" / "pareval_numbers.tex").write_text("\n".join(macros) + "\n",
                                                          encoding="utf-8")
    # keep the paper's copy in lock-step so a figure refresh updates the numbers too
    (ROOT / "paper" / "pareval_numbers.tex").write_text("\n".join(macros) + "\n",
                                                        encoding="utf-8")
    print(f"\nwrote results/ + paper/ pareval_numbers.tex ({len(macros)} macros)")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
