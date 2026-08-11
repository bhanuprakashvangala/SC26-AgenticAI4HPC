# Experiments

This directory contains the executable experiment entry points used to reproduce the artifact's core analyses.

Each experiment should be runnable independently and should read released inputs from `data/` and write machine-readable outputs to `results/`.

Planned experiment modules:

- `correctness_validation.py` — re-evaluate functional correctness under the released verification protocol.
- `serial_projection.py` — mechanically remove OpenMP directives and rerun the same correctness verification.
- `performance_spread.py` — summarize thread scaling across accepted programs.
- `candidate_selection.py` — compare reward-based candidate selection on the frozen candidate pools.

## Inputs

Released generations, verification records, timing measurements, and trusted references under `data/`.

## Outputs

CSV/JSON summaries under `results/`.

## Reproduction rule

Experiment scripts must not depend on manuscript source, LaTeX macros, or undocumented local files.
