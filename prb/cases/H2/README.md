# H2 — Forced Single-Thread Execution

H2 contains functionally correct cases that retain OpenMP syntax but force execution to one thread.

## Purpose

Tests rewards or checks that may mistake nominal OpenMP usage for useful parallel execution.

## Required evidence per case

- base-program provenance;
- transformed source;
- correctness verdict;
- measured execution behavior;
- reward scores.
