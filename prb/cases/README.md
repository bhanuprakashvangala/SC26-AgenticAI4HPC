# PRB Cases

This directory contains the concrete adversarial cases used by ParallelRewardBench.

Each attack-class subdirectory should contain its own `README.md` and the released source programs or case specifications needed to reproduce that class.

For every case, record:

- case identifier and attack class;
- base task/program provenance;
- construction or transformation applied;
- whether the case was executed or only modeled;
- correctness verdict;
- timing/performance measurements when applicable;
- reward scores produced by `prb/evaluate.py`.

The benchmark manifest should make it possible to trace every aggregate PRB number back to an individual released case.
