# Correct but Not Parallel

*What agent-generated OpenMP code pays to be race-free — an open harness for
measuring the correctness **and** the parallelism of LLM-generated HPC code.*

Submitted to the **1st International Workshop on Agentic AI for HPC
(AgenticAI4HPC 2026)**, co-located with SC26.

> **Thesis.** A correctness metric for parallel code that a *serial* program aces
> is not measuring parallel code. We give an agent a **distribution-aware
> differential verifier** (compile, sweep thread counts `{1,2,4,8}` with repeats
> against a trusted serial reference) and find that frontier models already pass
> it almost always — the silent-race failure mode it targets is largely absent,
> and an independent happens-before detector (LLVM Archer) agrees. The
> consequential finding is that **configuration-robust correctness is necessary
> but not sufficient, and gameable by construction**: among the 63 programs the
> verifier accepts as robust, measured self-speedup spans **0.33× to 7.9×** — one
> accepted program runs *slower* on eight threads than on one. Once frontier
> models make race-freedom cheap, "correct but not parallel" is where
> agent-generated HPC code now fails, and an output-only acceptance test cannot
> see it.

Built directly on **[ParEval](https://github.com/parallelcodefoundry/ParEval)**
(Nichols et al., HPDC'24), the standard benchmark for LLM-generated parallel code:
we reuse its function-completion prompts, its randomized-input drivers, and its
serial reference implementations as our trusted oracle.

---

## The two axes

1. **Correctness (necessary).** `R(G)` = agreement with the serial reference at
   *every* thread count and *every* repeat. Established by the differential
   verifier; cross-checked with LLVM Archer (ThreadSanitizer + OpenMP OMPT).
2. **Parallelism (the second axis).** Self-speedup `S(p) = T(1)/T(p)` and parallel
   efficiency `S(p)/p`, measured directly with ParEval's timing driver. `R(G)` is
   maximized by deleting every `#pragma` — so correctness alone cannot tell a
   scalable program from a serial one.

## Headline results

| Finding | Evidence |
|---|---|
| Correctness is largely solved at the frontier | L0 single-shot **82.8%** → L1 execute-once **87.9%** → L2 differential **87.9%** robust (Wilson CIs); L1==L2 on all 58 paired cells |
| The null is real, not an underpowered detector | **0 user-code races** across 63 robust programs under LLVM Archer, detector validated on a known race (27 generated-code stack frames) |
| The verifier's value returns as capability falls | open-weights panel: a 27B model ships a genuine shared-bin histogram race L1 certifies and L2 catches + repairs |
| **Robust ≠ parallel** (the reframe) | accepted programs span **0.33×–7.9×** self-speedup; one runs slower in parallel than serially |
| The "scan" failures are a spec artifact, not a bug | all models compute the reverse-indexed *suffix* sum (a valid reading of "reverse prefix sum"); excluding it, robust rate is **94.4%** |

## The tool-ablation

A "cell" is one `(task, model, condition, trial)`. The **same** propose→observe→
repair agent runs at three tool levels differing *only* in the observation the
verification tool returns:

| Condition | What the agent's verifier tool does |
|---|---|
| `single_shot` | no tool — one generation, scored as-is |
| `execute_once` | validate at **one** thread count (blind to races needing >1 thread), repair on failure |
| `differential_verify` | validate across the **thread sweep** vs. the serial reference, repair on failure |

Regardless of condition, the *final* program of every cell is scored by the full
differential verifier over the complete sweep — an outcome measure independent of
which tool the agent was allowed to use.

---

## What's in the artifact

```
harness/
  run_pareval_batch.py   MAIN runner: the propose->verify->repair agent loop;
                         batched, round-based agentic study over ParEval OpenMP
                         tasks. Pluggable backend (Azure VM or Nautilus pod) and
                         model router (Azure Foundry or NRP open-weights).
  measure_scaling.py     times every robust program across {1,2,4,8} threads with
                         ParEval's omp-driver -> results/scaling.jsonl
  analyze_scaling.py     scaling.jsonl -> speedup distribution + macros (the
                         "correct but not parallel" numbers)
  fig_scaling.py         the central figure: same-verdict/opposite-scaling + the
                         full accepted-program speedup distribution
  sanitizer_arm.py       cross-checks every robust program with LLVM Archer
                         (happens-before race detection) -> results/sanitizer.jsonl
  oversync.py            static classification of the synchronization idiom used
  cross_capability.py    frontier vs. open-weights panel on the race-prone subset
  make_figures.py        the correctness figure suite + all LaTeX macros
  make_tables.py         per-model + open-panel result tables
  figstyle.py            shared publication style (Okabe-Ito palette, Arial)
  azure_llm.py           keyless Azure Foundry access (Entra ID bearer tokens)
  nrp_llm.py             NRP/Nautilus open-weights access (OpenAI-compatible)
vm/
  sweep-driver.cc        validate-only thread-sweep driver ({1,2,4,8} x reps in
                         one process): "RUN t=.. r=.. valid=PASS|FAIL"
logs/pareval/            THE AGENT/LLM LOGS (written per-event at run time)
  generations/           one JSON PER LLM CALL: the service-emitted envelope
                         (request id, served model version, token usage, latency,
                         finish_reason), the exact messages, the verbatim response
  verify/                one JSON per verifier batch: raw runner stdout
  transcripts/           per-cell propose->verify->repair trajectory + verdicts
results/
  pareval_runs.jsonl     frontier study (one row per task x model x condition)
  pareval_open.jsonl     open-weights panel
  scaling.jsonl          per-program self-speedup across thread counts
  sanitizer.jsonl        Archer race-detection verdicts
  *_numbers.tex          LaTeX macros (paper inputs) — NO result is hand-typed
  figures/               the publication figure suite
paper/                   IEEEtran paper (main.tex, sections/, refs.bib, main.pdf)
```

## Reproduce

```powershell
python -m pip install -r requirements.txt

# 0) Get ParEval (excluded from this repo — it's a third-party checkout):
git clone https://github.com/parallelcodefoundry/ParEval external/ParEval

# 1) Model access. Azure Foundry (keyless, Entra ID): az login; put endpoint +
#    deployment names in .secrets/azure_openai.json. NRP open-weights (optional):
#    put a token from https://nrp.ai/llmtoken in .secrets/nrp_llm.json.

# 2) Run the agentic study (generate -> differential-verify -> repair -> certify).
python -u -m harness.run_pareval_batch --models gpt-5.4 gpt-5.2 `
  --conditions single_shot execute_once differential_verify

# 3) The second axis: time the robust programs, and cross-check with Archer.
$env:VERIFY_BACKEND="nautilus"      # or the Azure VM backend
python -m harness.measure_scaling ; python -m harness.analyze_scaling
python -m harness.sanitizer_arm --robust-only

# 4) Figures + tables + macros, then build the paper.
python -m harness.make_figures ; python -m harness.fig_scaling ; python -m harness.make_tables
cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main
```

The runner is **resumable**: it skips any cell already in the results file.

## Design notes

- **Serial reference as oracle** — sequential, so itself immune to the parallel
  hazards under test.
- **Repetition catches nondeterminism** — each thread count is validated multiple
  times over randomized inputs.
- **Two independent race instruments** — output-differential (the verifier) and
  happens-before (Archer) agree the accepted programs are race-free. Archer needs
  `clang -fsanitize=thread -fopenmp` + `LD_PRELOAD libarcher.so` + `setarch -R`
  (GCC's TSan silently misses OpenMP races).
- **Wall-time is directly measured** — the scaling finding does not depend on
  detector power.

## License / data

Uses only the open **ParEval** benchmark. No proprietary or personal data; no
credentials are stored in the repository (`.secrets/` is git-ignored).
