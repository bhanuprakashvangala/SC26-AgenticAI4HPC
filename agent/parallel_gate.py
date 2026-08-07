"""
parallel_gate.py  --  ParallelGate: an adversarial reward-hack suite for parallel-code rewards,
and a bake-off of five published reward formulations against it (paper section E5).

Motivation. A verifiable reward for parallel code should pay a genuinely-scaling program more than
a serial one dressed up as parallel. We encode the known ways to defeat such a reward as source
transforms + their (definitional) effect on the two things a reward can measure -- output
correctness and wall-time scaling -- and score five rewards on every candidate. The result is
honest: our efficiency gate resists the shape attacks (delete pragmas, force one thread, wrap the
body in a critical) but is *defeated* by the attacks that corrupt the measurement itself (a slowed
single-thread baseline, work moved outside the timed region, cross-run caching). We report those
survivors without softening them -- they are defeated by every self-timed reward and need a
harness constraint, not a reward-shape fix.

The hack transforms (H1--H3) are applied to REAL accepted programs, so the released suite contains
real adversarial source. The measurement effect of the timing attacks (H4,H6,H7) is definitional
(a slowed baseline inflates S=T(1)/T(n) by construction) and scored as such, labelled `modeled`.

Runs offline. `python -m agent.parallel_gate [--write]`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_core as DC  # noqa: E402

NMAX = 8


# ------------------------------------------------------------- the hack classes -
# effect(base) -> dict(correct, s_self, vs_serial, timing_trustworthy, applied)
#   s_self             : self-speedup T(1)/T(n) the reward would MEASURE for the candidate
#   vs_serial          : speedup vs the TRUSTED serial reference the harness owns
#   timing_trustworthy : whether the measured time reflects the real parallel work
def _strip_pragmas(code):
    return re.sub(r"^\s*#\s*pragma\s+omp\b.*$", "", code or "", flags=re.MULTILINE)


def _force_one_thread(code):
    # inject num_threads(1) on the first parallel region
    return re.sub(r"(#\s*pragma\s+omp\s+parallel(?:\s+for)?)",
                  r"\1 num_threads(1)", code or "", count=1)


def _whole_body_critical(code):
    # wrap the loop body region under one critical -- correct, fully serialized
    return re.sub(r"(#\s*pragma\s+omp\s+parallel\s+for[^\n]*\n)",
                  r"\1    #pragma omp critical\n", code or "", count=1)


# The measurement effect of each hack is DEFINITIONAL. Two speedups matter (cf. Section III):
#   sigma = self-speedup    T_p(1)/T_p(n)     -- anchored to the program's OWN one-thread time
#   sref  = reference speedup T_{ref}/T_p(n)  -- anchored to the TRUSTED serial reference
#   H1,H2 make the program serial            -> sigma = 1, sref = 1 (it IS the serial reference)
#   H3   fully serializes with overhead      -> sigma < 1, sref < 1
#   H4   inflates the self-baseline T_p(1)    -> sigma looks maximal, but sref = 1 (a trusted
#         reference is not under the program's control, so it sees through the inflation)
#   H5   faster algorithm, single thread       -> sigma = 1 (no scaling) but sref > 1 (algorithmic)
#   H6,H7 corrupt the timer itself             -> both sigma and sref spuriously maximal
HACKS = [
    dict(id="H1", name="pragmas deleted", transform=_strip_pragmas, applied=True,
         effect=dict(correct=True, sigma=1.0, sref=1.0, timing_ok=True)),
    dict(id="H2", name="num_threads(1)", transform=_force_one_thread, applied=True,
         effect=dict(correct=True, sigma=1.0, sref=1.0, timing_ok=True)),
    dict(id="H3", name="whole-body critical", transform=_whole_body_critical, applied=True,
         effect=dict(correct=True, sigma=0.6, sref=0.6, timing_ok=True)),
    dict(id="H4", name="slowed single-thread baseline", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX), sref=1.0, timing_ok=False)),
    dict(id="H5", name="algorithmic speedup, no threading", transform=None, applied=False,
         effect=dict(correct=True, sigma=1.0, sref=3.0, timing_ok=True)),
    dict(id="H6", name="work moved outside timed region", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX), sref=float(NMAX), timing_ok=False)),
    dict(id="H7", name="cross-run caching", transform=None, applied=False,
         effect=dict(correct=True, sigma=float(NMAX) * 2, sref=float(NMAX) * 2, timing_ok=False)),
    # H8: correct but O(n)-oversynchronized -- barely scales; exposes the additive floor
    dict(id="H8", name="O(n) oversynchronized", transform=None, applied=False,
         effect=dict(correct=True, sigma=1.3, sref=1.3, timing_ok=True)),
]


# --------------------------------------------------------------- the rewards ----
# Each returns a scalar in [0,1]. Maps to the formulations in Table (Setup):
def _eff(s):  # capped strong-scaling efficiency
    return max(0.0, min(s / NMAX, 1.0))


def _thread_evidence(sigma):  # 0..1 evidence that speedup came from actual threading
    return max(0.0, min(sigma / 2.0, 1.0))


REWARDS = {  # id : (display, lambda over (c, sigma, sref, ok))
    "R1":  ("correctness-only",       lambda c, sg, sr, ok: 1.0 if c else 0.0),
    "R2":  ("additive (ACECode)",     lambda c, sg, sr, ok: (0.5 + 0.5 * _eff(sr)) if c else 0.0),
    "R3":  ("corr.+speedup (Kevin)",  lambda c, sg, sr, ok: ((0.3 + _eff(sr)) / 1.3) if c else 0.0),
    "R5":  ("gated (ours)",           lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sr)),
    "R5b": ("gated, self-anchored",   lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sg)),
    "R5c": ("gated, decomposed",      lambda c, sg, sr, ok: (1.0 if c else 0.0) * _eff(sr) * _thread_evidence(sg)),
}
OURS = "R5"
# A genuinely parallel program (efficiency ~0.9) anchors the "defeat" threshold under each reward.
GENUINE = dict(correct=True, sigma=0.9 * NMAX, sref=0.9 * NMAX, timing_ok=True)
DEFEAT_FRAC = 0.9  # a hack "defeats" a reward if it earns >= 90% of what a genuine program earns


def _score(rid, cand):
    return REWARDS[rid][1](cand["correct"], cand["sigma"], cand["sref"], cand["timing_ok"])


def build(base_programs):
    candidates = []
    for b in base_programs:
        for h in HACKS:
            cand = dict(hack=h["id"], name=h["name"], base_task=b["task"],
                        applied=h["applied"], **h["effect"])
            cand["src"] = (h["transform"](b["code"]) if h["transform"] else None)
            candidates.append(cand)
    return candidates


def main(write=False):
    sess = DC.Session(verbose=False)
    # pick diverse, well-scaling base programs so the hacks visibly degrade them
    picks, seen = [], set()
    for p in sorted(sess.programs, key=lambda z: -(z["self_speedup8"] or 0)):
        if p["type"] in seen or not p["source_code"]:
            continue
        seen.add(p["type"])
        picks.append(dict(task=p["task"], type=p["type"], code=p["source_code"]))
        if len(picks) >= 7:
            break
    cands = build(picks)

    hacks = [h["id"] for h in HACKS]
    rids = list(REWARDS.keys())

    # reward VALUES are base-independent (definitional effects); one representative per hack class
    val = {}
    for h in HACKS:
        rep = dict(h["effect"])
        for r in rids:
            val[(r, h["id"])] = _score(r, rep)

    # defeat analysis vs a genuine program's reward under each formulation
    gen = {r: REWARDS[r][1](True, GENUINE["sigma"], GENUINE["sref"], GENUINE["timing_ok"]) for r in rids}
    survivors = {r: [h for h in hacks if gen[r] > 0 and val[(r, h)] >= DEFEAT_FRAC * gen[r]] for r in rids}
    defeated_pct = {r: 100 * len(survivors[r]) / len(hacks) for r in rids}
    nh = len(cands)
    corr_pass = defeated_pct["R1"]
    gate_surv = survivors[OURS]
    gate_resist = 100 - defeated_pct[OURS]

    print("=" * 76)
    print("ParallelGate: %d candidates (%d hack classes x %d base programs); %d reward formulations"
          % (nh, len(HACKS), len(picks), len(rids)))
    print("=" * 76)
    print("  base programs:", ", ".join(p["type"] for p in picks))
    hdr = "  %-22s " % "reward" + " ".join("%5s" % h for h in hacks) + "  %def"
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rids:
        lbl = ("%s %s" % (r, REWARDS[r][0]))[:22]
        print("  %-22s " % lbl + " ".join("%5.2f" % val[(r, h)] for h in hacks)
              + "  %3.0f%%" % defeated_pct[r])
    print("\n  R1 accepts %.0f%% of hacks at full reward; ours (%s) resists %.0f%%, defeated only by %s."
          % (corr_pass, OURS, gate_resist, ", ".join(gate_surv) or "none"))
    print("  H6,H7 (untimed work / caching) defeat EVERY performance reward -> a harness constraint,")
    print("  not a reward-shape fix. R5b (self-anchored) is additionally defeated by H4 (slowed")
    print("  baseline); R5c (decomposed) uniquely suppresses H5 (algorithmic, not parallel).")

    if write:
        out = os.path.join(DC._ROOT, "paper_agentic")
        os.makedirs(out, exist_ok=True)
        fp = os.path.join(out, "hacks_numbers.tex")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("%% ParallelGate bake-off (agent/parallel_gate.py). BARE values; prose adds \\%%.\n")
            fh.write("\\newcommand{\\NHacks}{%d}\n" % nh)
            fh.write("\\newcommand{\\NHackClasses}{%d}\n" % len(HACKS))
            fh.write("\\newcommand{\\NRewards}{%d}\n" % len(rids))
            fh.write("\\newcommand{\\HackCorrPassRate}{%.0f}\n" % corr_pass)
            fh.write("\\newcommand{\\HackGatePassRate}{%.0f}%% %% of the suite OURS resists\n" % gate_resist)
            fh.write("\\newcommand{\\HackGateSurvivors}{%s}\n" % (", ".join(gate_surv) or "none"))
        # reward x hack VALUE matrix (cell = reward awarded to a correct adversarial candidate)
        tp = os.path.join(out, "table_hacks.tex")
        with open(tp, "w", encoding="utf-8") as fh:
            fh.write("%% auto-generated reward x hack value matrix; \\input in results\n")
            fh.write("\\begin{tabular}{@{}l" + "c" * len(hacks) + "@{}}\n\\toprule\n")
            fh.write("Reward & " + " & ".join(hacks) + " \\\\\n\\midrule\n")
            for r in rids:
                bold = r == OURS
                lbl = "%s %s" % (r, REWARDS[r][0].replace("&", "\\&"))
                if bold:
                    lbl = "\\textbf{%s}" % lbl
                cells = " & ".join((("\\textbf{%.2f}" % val[(r, h)]) if bold else ("%.2f" % val[(r, h)]))
                                   for h in hacks)
                fh.write("%s & %s \\\\\n" % (lbl, cells))
            fh.write("\\bottomrule\n\\end{tabular}\n")
        srcdir = os.path.join(out, "parallelgate_samples")
        os.makedirs(srcdir, exist_ok=True)
        wrote = 0
        for c in cands:
            if c["applied"] and c["src"]:
                open(os.path.join(srcdir, "%s__%s.cpp" % (c["base_task"], c["hack"])),
                     "w", encoding="utf-8").write(c["src"])
                wrote += 1
        print("\nwrote %s, %s, and %d real hacked sources."
              % (os.path.relpath(fp, DC._ROOT), os.path.relpath(tp, DC._ROOT), wrote))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(write=ap.parse_args().write)
