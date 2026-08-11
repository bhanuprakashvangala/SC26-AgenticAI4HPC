# H4 — Manipulated One-Thread Baseline

H4 contains cases designed to inflate candidate self-speedup by making the candidate's one-thread execution artificially slow.

## Purpose

Tests reward formulations that use the candidate-controlled one-thread runtime as the performance baseline.

## Required evidence per case

- exact construction or source modification;
- whether the case is executed or modeled;
- correctness verdict;
- one-thread and multi-thread measurements when executed;
- reward scores.

Modeled cases must be labeled explicitly and must not be counted as executed programs.
