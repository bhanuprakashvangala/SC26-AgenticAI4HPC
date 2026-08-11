# H6 — Work Outside the Timed Region

H6 contains cases that make the measured runtime look artificially small by moving useful work outside the region observed by the timer.

## Purpose

Tests timing-integrity assumptions rather than only reward arithmetic.

## Required evidence per case

- exact source or harness construction;
- whether the case is executed or modeled;
- correctness verdict;
- timing protocol and raw measurement evidence when executed;
- reward scores.

Modeled cases must be labeled explicitly and must not be counted as executed programs.
