# Execution Harness

This directory contains the code that generates, compiles, verifies, times, and cross-checks OpenMP candidates.

## Responsibilities

- prepare ParEval tasks and trusted serial references;
- call configured model backends when full regeneration is requested;
- compile generated programs with the documented compiler flags;
- run correctness verification across the configured thread sweep and repetitions;
- collect timing measurements;
- run sanitizer/race-detection cross-checks;
- write raw records consumed by `experiments/`.

## Inputs

Task definitions, model configuration, execution-backend configuration, and optional credentials supplied outside the repository.

## Outputs

Raw generation, verification, timing, and sanitizer records that are later normalized into `data/` and analyzed by `experiments/`.

## Important

The harness defines measurement and verification behavior. Analysis code should not silently reimplement or weaken these checks.
