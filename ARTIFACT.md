# Artifact Description and Reproduction Guide

This document describes the reproducibility artifact accompanying the SC26 AgenticAI4HPC submission **Correct but Not Parallel: Performance-Aware Verifiable Rewards for AI-Generated HPC Code**.

## 1. Scope

The artifact supports reproduction of the paper's core analyses from frozen experimental data and, when the required external services are available, regeneration of the underlying model outputs and execution measurements.

The paper studies LLM-generated OpenMP programs after functional correctness has been established. It evaluates whether correctness alone captures parallel behavior, measures scaling among accepted programs, compares reward formulations, and evaluates them with **ParallelRewardBench (PRB)**.

The current PRB release contains the adversarial cases used in the paper. PRB is designed as an extensible benchmark; additional candidates and attack classes may be added in later releases as new reward formulations and failure modes are identified.

## 2. Artifact contents

### Experiment and analysis code

- `harness/run_pareval_batch.py` — generation and verification pipeline.
- `harness/measure_scaling.py` — measures accepted programs across thread counts.
- `harness/analyze_scaling.py` — derives scaling statistics from frozen timing data.
- `harness/sanitizer_arm.py` — Archer/ThreadSanitizer cross-check.
- `agent/diag_core.py` — offline loader for the frozen accepted-program pool.
- `agent/compute_numbers.py` — derives paper statistics from the frozen data.
- `agent/parallel_reward_bench.py` — public entry point for **ParallelRewardBench (PRB)**.
- `agent/make_paper_figs.py` — publication figure generation from frozen measurements.

The historical PRB implementation remains in `agent/parallel_gate.py` so previously generated artifact outputs remain reproducible; new users should invoke `agent.parallel_reward_bench`.

### Frozen data and provenance

- `results/pareval_runs.jsonl` — verification results.
- `results/scaling.jsonl` — per-thread timing and scaling measurements.
- `results/sanitizer.jsonl` — race-detection results.
- `results/oversync.jsonl` — synchronization-pattern analysis.
- `logs/pareval/generations/` — model prompts, responses, and extracted generated source.
- `logs/pareval/verify/` and `logs/pareval/transcripts/` — verification output and generation/repair trajectories when present.

These files allow the main analyses to be inspected without contacting the original model services.

## 3. Reproducibility tracks

### Track A: frozen-data analysis

This is the recommended path for artifact evaluation because it avoids external model APIs and reproduces the analysis from the released measurements.

Requirements:

- Python 3.10 or newer
- packages in `requirements.txt`

Run:

```bash
python -m pip install -r requirements.txt
python -m harness.analyze_scaling
python -m agent.compute_numbers
python -m agent.parallel_reward_bench
python -m agent.make_paper_figs
```

Expected outputs include scaling summaries, reward-comparison output, and regenerated figures. Scripts that support `--write` can also emit derived LaTeX macros or benchmark tables.

### Track B: correctness and sanitizer inspection

This track inspects the frozen correctness and race-detection results and can rerun analysis scripts without model generation.

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

The artifact does not include credentials. Full regeneration may also differ slightly from the frozen data because hosted model versions and shared-system performance can change over time.

## 4. Core measurements

For a generated candidate `p` evaluated at `n` threads:

- `c(p)` is the binary correctness indicator.
- `S_p(n) = T_ref / T_p(n)` is speedup relative to the trusted sequential reference.
- `Sigma_p(n) = T_p(1) / T_p(n)` is candidate self-speedup.
- the performance-aware reward studied in the paper is

  `R_perf(p) = c(p) * min(S_p(n)/n, 1)`.

The artifact focuses on runtime speedup because the current experiments target OpenMP thread scaling. The reward function can be extended to incorporate additional measurable objectives, but those extensions are outside the current experimental scope.

## 5. Serial-projection analysis

The serial projection mechanically removes OpenMP directives from a generated candidate while leaving the remaining computation unchanged. It is used to test whether the correctness signal changes when explicit parallelization is removed.

The paper reports both the correctness behavior of the projected programs and the corresponding change in performance-aware reward. This transformation is an analysis step applied to generated code; the model is not asked to generate a separate serial implementation.

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

PRB is not tied to the proposed reward. Its purpose is to provide a common adversarial evaluation for multiple parallel-code reward formulations. The current release should be treated as the benchmark version evaluated by the paper, not as a closed or final taxonomy.

## 7. Expected reproducibility boundaries

The artifact distinguishes three kinds of reproducibility:

1. **Deterministic analysis reproduction:** recomputing statistics and tables from released JSON/JSONL files.
2. **Measurement reproduction:** rerunning compiled programs on a compatible OpenMP system. Absolute runtimes may vary by machine, but the protocol is fixed.
3. **Generation reproduction:** rerunning hosted models. Exact text may vary because hosted model implementations and service versions can change.

For this reason, the released generation logs and frozen timing data are the provenance record for the paper's reported results.

## 8. Credentials and sensitive data

No credentials are committed. `.secrets/`, private-key extensions, API-key-like names, and common authentication files are excluded by `.gitignore`.

The artifact uses benchmark-generated code and experiment metadata; it contains no personal participant data.

## 9. Third-party software

The study builds on the ParEval benchmark. The repository may include a vendored checkout for self-contained reproduction; otherwise obtain ParEval from its upstream repository. Third-party code remains subject to its original license.

## 10. Paper build

The current manuscript sources are under `paper_agentic/`. Once the final terminology and figures are frozen, build using the IEEEtran/BibTeX sequence documented in the paper directory or the root README.

The artifact should be cited using the repository URL given in the paper's Artifact Availability appendix.
