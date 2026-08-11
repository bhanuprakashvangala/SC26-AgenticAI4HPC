# Released Data

This directory contains the frozen data needed to reproduce the artifact without model access.

The goal is to separate immutable experimental evidence from analysis code and derived summaries.

## Subdirectories

- `generations/` — prompts, model responses, and extracted generated source.
- `accepted/` — normalized metadata for candidates that pass the released correctness criteria.
- `timings/` — per-thread runtime measurements and timing metadata.
- `projections/` — serial-projection source/verdict records produced by executable re-verification.
- `sanitizer/` — Archer/ThreadSanitizer cross-check outputs.

## Data policy

Every released record should include enough identifiers to trace it to its task, model, condition, and source program. Derived statistics belong in `results/`, not here.
