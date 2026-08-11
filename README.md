# SC26 AgenticAI4HPC Artifact

A reproducibility artifact for evaluating correctness and parallel performance in LLM-generated OpenMP code, including the **ParallelRewardBench (PRB)** adversarial benchmark.

## What is included

This repository contains:

- generation and verification code for OpenMP candidates;
- frozen generation logs and extracted source programs;
- correctness, timing, scaling, and race-detection measurements;
- reward-analysis utilities;
- ParallelRewardBench (PRB), an extensible adversarial benchmark for parallel-code reward functions;
- scripts for reproducing the released analyses from frozen data.

## Repository layout

```text
harness/                  generation, verification, timing, and analysis pipeline
agent/                    offline analysis and PRB utilities
vm/                       OpenMP thread-sweep execution driver
logs/pareval/             generation and verification provenance
results/                   frozen measurements and derived outputs
external/ParEval/          vendored ParEval checkout when present
ARTIFACT.md                detailed reproduction and provenance guide
```

## Core frozen data

```text
results/pareval_runs.jsonl     correctness and verification records
results/scaling.jsonl          measured per-thread runtimes and scaling results
results/sanitizer.jsonl        Archer/ThreadSanitizer cross-checks
results/oversync.jsonl         synchronization-pattern analysis
logs/pareval/generations/      prompts, model responses, and extracted source
```

The accepted timing pool contains 63 generated programs. Frozen data are preserved so the main analyses can be rerun without calling a model API.

## Quick reproduction

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Recompute the main frozen-data analyses:

```bash
python -m harness.analyze_scaling
python -m agent.compute_numbers
python -m agent.parallel_reward_bench
```

No model credentials are required for these analysis steps.

## Full regeneration

Full regeneration additionally requires model access and an execution backend with GCC/OpenMP.

```bash
python -u -m harness.run_pareval_batch \
  --models gpt-5.4 gpt-5.2 \
  --conditions single_shot execute_once differential_verify

python -m harness.measure_scaling
python -m harness.analyze_scaling
python -m harness.sanitizer_arm --robust-only
```

Hosted-model outputs and absolute runtimes can vary over time or across machines, so the released generation logs and timing files are the provenance record for this artifact release.

## Measurements

For candidate program `p` at `n` threads:

- correctness: `c(p) in {0,1}`
- reference speedup: `S_p(n) = T_ref / T_p(n)`
- self-speedup: `Sigma_p(n) = T_p(1) / T_p(n)`
- performance-aware reward: `R_perf(p) = c(p) * min(S_p(n)/n, 1)`

The current artifact uses runtime speedup as the measured performance signal.

## ParallelRewardBench

PRB evaluates functionally correct adversarial candidates that stress whether a reward measures useful parallel behavior rather than only successful output or a manipulable timing signal.

The current release covers eight classes:

- **H1** — OpenMP directives removed
- **H2** — forced single-thread execution
- **H3** — useful work serialized with `critical`
- **H4** — manipulated one-thread baseline
- **H5** — faster sequential algorithm without meaningful thread scaling
- **H6** — useful work outside the timed region
- **H7** — cross-run caching/reuse
- **H8** — correct but heavily synchronized execution

PRB is designed to be extensible. Additional candidates and attack classes can be added as new reward formulations and failure modes are studied.

## Third-party benchmark

The experiments build on **ParEval** (Nichols et al., HPDC 2024), using its OpenMP tasks, randomized-input drivers, and trusted serial references. Third-party code remains subject to its original license.

## Credentials and data

No API keys, tokens, or personal datasets are committed. `.secrets/` and common credential formats are excluded by `.gitignore`.

See [`ARTIFACT.md`](ARTIFACT.md) for the detailed reproduction guide.
