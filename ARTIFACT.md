# Artifact Description and Reproduction Guide

## 1. Scope

This repository is a standalone reproducibility artifact for experiments on correctness and parallel performance in LLM-generated OpenMP code. It contains frozen model generations, execution measurements, verification outputs, reward analyses, and ParallelRewardBench (PRB).

The artifact is designed so that the main analyses can be rerun from released data without requiring model access. Full regeneration is also supported when the required external services and execution environment are available.

## 2. Artifact contents

### Experiment and analysis code

- `harness/run_pareval_batch.py` — generation and verification pipeline.
- `harness/measure_scaling.py` — measures accepted programs across thread counts.
- `harness/analyze_scaling.py` — derives scaling statistics from frozen timing data.
- `harness/sanitizer_arm.py` — Archer/ThreadSanitizer cross-check.
- `agent/diag_core.py` — offline loader for the frozen accepted-program pool.
- `agent/compute_numbers.py` — derives summary statistics from the frozen data.
- `agent/parallel_reward_bench.py` — public entry point for ParallelRewardBench (PRB).

### Frozen data and provenance

- `results/pareval_runs.jsonl` — correctness and verification results.
- `results/scaling.jsonl` — per-thread timing and scaling measurements.
- `results/sanitizer.jsonl` — race-detection results.
- `results/oversync.jsonl` — synchronization-pattern analysis.
- `logs/pareval/generations/` — model prompts, responses, and extracted generated source.
- `logs/pareval/verify/` — raw verification output when present.
- `logs/pareval/transcripts/` — generation/repair trajectories when present.

These files are the provenance record for the released artifact.

## 3. Reproducibility tracks

### Track A: frozen-data analysis

This is the recommended path because it requires no model API and uses the released measurements directly.

Requirements:

- Python 3.10 or newer
- packages in `requirements.txt`

Run:

```bash
python -m pip install -r requirements.txt
python -m harness.analyze_scaling
python -m agent.compute_numbers
python -m agent.parallel_reward_bench
```

### Track B: correctness and sanitizer inspection

```bash
python -m harness.analyze_pareval
python -m harness.make_tables
```

A full Archer rerun requires the compiler/runtime setup described by `harness/sanitizer_arm.py` and is platform dependent.

### Track C: full experiment regeneration

Full regeneration additionally requires:

- access to the configured model endpoints;
- an execution backend with GCC and OpenMP;
- the ParEval task suite;
- environment-specific authentication configured outside the repository.

Example:

```bash
python -u -m harness.run_pareval_batch \
  --models gpt-5.4 gpt-5.2 \
  --conditions single_shot execute_once differential_verify

python -m harness.measure_scaling
python -m harness.analyze_scaling
python -m harness.sanitizer_arm --robust-only
```

The artifact does not include credentials. Hosted-model outputs and absolute performance measurements may differ when regenerated later or on different hardware.

## 4. Core measurements

For a generated candidate `p` evaluated at `n` threads:

- `c(p)` is the binary correctness indicator.
- `S_p(n) = T_ref / T_p(n)` is speedup relative to the trusted sequential reference.
- `Sigma_p(n) = T_p(1) / T_p(n)` is candidate self-speedup.
- `R_perf(p) = c(p) * min(S_p(n)/n, 1)` is the performance-aware reward used by the artifact analysis.

The current artifact uses runtime speedup as the performance signal.

## 5. Serial projection

The serial projection mechanically removes OpenMP directives from a generated candidate while leaving the remaining computation unchanged. It is used to examine whether correctness changes when explicit parallelization is removed and how the corresponding performance signal changes.

The transform is applied to generated code; the model is not asked to generate a separate serial implementation.

## 6. ParallelRewardBench (PRB)

PRB currently evaluates eight classes of functionally correct adversarial behavior:

| Class | Construction | Primary target |
|---|---|---|
| H1 | OpenMP directives removed | correctness-only evaluation |
| H2 | forced one-thread execution | static/nominal parallelism |
| H3 | useful work serialized by `critical` | nominal parallelism |
| H4 | candidate one-thread path slowed | self-speedup baseline |
| H5 | better sequential algorithm, little thread scaling | reference-speedup baseline |
| H6 | useful work outside timed region | timing integrity |
| H7 | cross-run caching/reuse | repeated timing |
| H8 | correct but heavily synchronized | weak parallel performance |

PRB is not tied to one reward formulation. It provides a common adversarial evaluation for parallel-code rewards. The benchmark is designed to be extended with additional candidates and attack classes as new failure modes are identified.

## 7. Reproducibility boundaries

The artifact distinguishes three levels:

1. **Analysis reproduction:** recompute statistics from released JSON/JSONL files.
2. **Measurement reproduction:** rerun generated programs on a compatible OpenMP system; absolute runtimes can change across machines.
3. **Generation reproduction:** rerun hosted models; exact generated source can change as services evolve.

The frozen generation logs and timing data therefore serve as the stable record for this artifact release.

## 8. Credentials and sensitive data

No credentials are committed. `.secrets/`, private-key extensions, API-key-like names, and common authentication files are excluded by `.gitignore`.

The artifact contains benchmark-generated code and experiment metadata only; it contains no personal participant data.

## 9. Third-party software

The experiments build on the ParEval benchmark. The repository may include a vendored checkout for self-contained reproduction; otherwise obtain ParEval from its upstream repository. Third-party code remains subject to its original license.
