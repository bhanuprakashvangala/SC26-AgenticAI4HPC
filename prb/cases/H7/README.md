# H7 — Cross-Run Caching

H7 contains cases that reuse previously computed results so repeated timing can report work that was not recomputed during the measured run.

## Purpose

Tests repeated-measurement integrity and whether the harness prevents cross-run reuse from inflating performance.

## Required evidence per case

- exact source or execution construction;
- whether the case is executed or modeled;
- correctness verdict;
- repetition/timing protocol and raw measurements when executed;
- reward scores.

Modeled cases must be labeled explicitly and must not be counted as executed programs.
