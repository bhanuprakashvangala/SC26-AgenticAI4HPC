# H3 — Whole-Body Serialization

H3 contains functionally correct cases whose useful loop work is effectively serialized, for example by wrapping the loop body in a `critical` region.

## Purpose

Tests whether a reward distinguishes nominal parallel structure from useful parallel execution.

## Required evidence per case

- base-program provenance;
- transformed source;
- correctness verdict;
- timing/scaling measurements;
- reward scores.
