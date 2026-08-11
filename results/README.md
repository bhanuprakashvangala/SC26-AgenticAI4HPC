# Derived Results

This directory contains machine-readable outputs produced from the released data by the artifact's analysis and benchmark scripts.

Expected outputs include:

- `correctness.csv` — correctness-validation summary;
- `projection.csv` — serial-projection outcomes;
- `scaling.csv` — scaling statistics for accepted candidates;
- `selection.csv` — reward-based candidate-selection comparison;
- `prb_scores.csv` — ParallelRewardBench reward matrix and attack outcomes.

Raw evidence belongs in `data/`; this directory is for reproducible derived results only.

Every result file should be regenerable from documented commands in the root README and the relevant experiment README.
