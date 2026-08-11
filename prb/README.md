# ParallelRewardBench (PRB)

This directory contains the standalone adversarial benchmark for evaluating parallel-code reward functions.

PRB is independent of any single proposed reward. Each released case must have an auditable construction, source program or generator, correctness status, performance metadata, and reward scores.

## Structure

- `cases/` — concrete adversarial cases grouped by attack class.
- `attacks.py` — definitions or generators for attack classes.
- `rewards.py` — reward formulations evaluated by the benchmark.
- `evaluate.py` — runs the benchmark and writes machine-readable scores.

## Current attack classes

H1–H8 cover removed or neutralized parallelism, baseline manipulation, timing-measurement attacks, and weak scaling.

## Outputs

Benchmark results should be written to `results/prb_scores.csv` or an equivalent machine-readable file.

## Reproducibility rule

A case must not be counted as an executed program unless its source and execution/verification evidence are released. Modeled cases must be clearly labeled as modeled rather than measured.
