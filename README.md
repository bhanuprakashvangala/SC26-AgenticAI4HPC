# SC26 AgenticAI4HPC Artifact

Artifact for **Correct but Not Parallel: Performance-Aware Verifiable Rewards for AI-Generated HPC Code**, submitted to the 1st International Workshop on Agentic AI for HPC (AgenticAI4HPC 2026), co-located with SC26.

## What this artifact supports

This repository contains the code, frozen measurements, generated-program logs, and analysis scripts used to support four claims in the paper:

1. **Correctness alone does not observe parallel performance.** We mechanically remove OpenMP directives from accepted generated programs and re-evaluate them (the serial-projection experiment).
2. **Correct programs can have very different parallel behavior.** The accepted timing pool spans **0.33×–7.91×** eight-thread self-speedup.
3. **A performance-aware reward can distinguish correct candidates.** The paper evaluates a reward based on correctness and measured reference speedup, together with comparison formulations and ablations.
4. **ParallelRewardBench (PRB)** stress-tests parallel-code reward functions with adversarial, functionally correct candidates that target useful parallelism, performance baselines, and timing measurement.

The current PRB release contains the cases evaluated in the paper and is intentionally extensible: new candidates and attack classes can be added as new reward formulations and failure modes are studied.

## Repository layout

```text
harness/                  generation, verification, timing, and analysis pipeline
agent/                    offline reward/PRB analysis helpers
vm/                       thread-sweep execution driver
logs/pareval/             model-generation and verification provenance
results/                   frozen measurements and derived result files
paper_agentic/             current paper sources and publication figures
external/ParEval/          vendored ParEval checkout when present
ARTIFACT.md                detailed reproduction and provenance guide
```

### Core frozen data

The main analyses use the following files:

```text
results/pareval_runs.jsonl     generated-program correctness/verification records
results/scaling.jsonl          measured per-thread runtimes and scaling results
results/sanitizer.jsonl        Archer/ThreadSanitizer cross-checks
results/oversync.jsonl         synchronization-pattern analysis
logs/pareval/generations/      exact prompts, model responses, and extracted source
```

The paper's accepted timing pool contains **63 programs**. The repository preserves the generation records and frozen measurements so analyses can be rerun without calling any model API.

## Reproduction levels

### A. Reproduce analyses from frozen data (recommended)

This path requires no model access and is the easiest way to reproduce the paper's reported analyses.

```bash
python -m pip install -r requirements.txt
python -m harness.analyze_scaling
python -m agent.compute_numbers
python -m agent.parallel_gate
```

`agent/parallel_gate.py` is the historical implementation filename used by the current artifact; in the paper the benchmark is named **ParallelRewardBench (PRB)**. The file will be renamed in a future artifact revision without changing the benchmark semantics.

### B. Regenerate publication figures

```bash
python -m agent.make_paper_figs
```

The figure scripts read the frozen result files; they do not require model access.

### C. Re-run generation and verification

Full regeneration requires access to the configured model endpoints and an execution backend with GCC/OpenMP. Credentials are never stored in this repository.

```bash
python -u -m harness.run_pareval_batch \
  --models gpt-5.4 gpt-5.2 \
  --conditions single_shot execute_once differential_verify
```

Then re-run timing and correctness cross-checks:

```bash
python -m harness.measure_scaling
python -m harness.analyze_scaling
python -m harness.sanitizer_arm --robust-only
```

See `ARTIFACT.md` for environment details and the distinction between frozen-data reproduction and full experiment regeneration.

## Measurement conventions

For candidate program `p` at `n` threads:

- correctness: `c(p) ∈ {0,1}`
- reference speedup: `S_p(n) = T_ref / T_p(n)`
- self-speedup: `Sigma_p(n) = T_p(1) / T_p(n)`
- performance-aware reward used in the paper: `R_perf(p) = c(p) * min(S_p(n)/n, 1)`

The paper focuses on runtime speedup as the measured performance signal. The reward formulation is not inherently restricted to runtime and can be extended with additional measurable objectives in future work.

## ParallelRewardBench

PRB evaluates reward functions on functionally correct adversarial candidates. The current release covers eight classes:

- H1: OpenMP directives removed
- H2: forced single-thread execution
- H3: useful work serialized with `critical`
- H4: manipulated one-thread baseline
- H5: faster sequential algorithm without meaningful thread scaling
- H6: work moved outside the timed region
- H7: cross-run caching/reuse
- H8: correct but heavily synchronized parallel execution

H1–H3 primarily test whether a reward distinguishes useful parallel execution; H4–H5 test the chosen performance baseline; H6–H7 test measurement integrity; H8 tests weak but valid parallel execution.

## Third-party benchmark

The study builds on **ParEval** (Nichols et al., HPDC 2024) and reuses its OpenMP function-completion tasks, randomized-input drivers, and trusted serial references. If the vendored checkout is not present, obtain it from the upstream ParEval repository before full regeneration.

## Credentials and privacy

No API keys, tokens, private user data, or personal datasets are required for frozen-data analysis. `.secrets/` and common key formats are excluded by `.gitignore`.

## Paper and artifact status

The repository is being prepared as the public reproducibility artifact for the SC26 workshop submission. The frozen data and scripts correspond to the experiments reported in the paper; manuscript wording and figure labels may continue to receive non-substantive cleanup before camera-ready release.
