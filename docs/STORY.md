# The whole thing, in plain words

*How the pieces fit, why the problem matters, what every experiment does, which
benchmarks we use, what is genuinely new, and exactly how to run it.*

---

## 1. The one-paragraph version

AI agents now write parallel HPC code and **check their own work** with an execution
tool: generate, run the checker, ship what passes. That same checker signal is what the
newest models are *trained* on — reinforcement learning from a verifier's reward instead
of from human preference (this is what "RLVR" and "GRPO" mean). So the checker is the
real definition of "good code." We ask: **what must that checker measure?** The obvious
answer is correctness. We show correctness is the *wrong sole answer* for parallel code,
for two reasons: (1) it's **saturated** — strong models are already correct, so a fancier
correctness checker adds nothing; and (2) it's **gameable** — a program with the
parallelism deleted (plain serial code) passes every correctness check perfectly, so a
reward built on correctness is *maximized by removing the parallelism HPC exists for*.
The fix is a **two-axis reward**: correct **and** actually fast. We build that as a second
"gate" in a verification-guided agent (this is the VG-RLPT idea), and we show — on our own
data, with live models, and with real RL — that adding the performance gate recovers the
speed a correctness-only reward throws away.

---

## 2. The problem (why this matters)

When an agent writes a `for` loop with OpenMP, whether the answer is *right* can depend on
**how many threads run it**. The textbook example, which shows up throughout our results:

> An agent is asked to build a histogram — count how many values fall in each bin. It
> writes a parallel loop where every thread bumps a shared counter. It runs it once, sees
> a sensible-looking histogram, and ships. **It's wrong.** Two threads bump the same
> counter at the same time and one update is lost — a *data race*. But on **one** thread
> there's no one to collide with, so it's exactly right. And even on more threads the lost
> counts can be small enough to look fine.

The bug is invisible to the exact check the agent performed (run it once). This is the
gap the field has been chasing: correctness that only shows up when you change the number
of threads.

## 3. The idea that ties it together: the checker *is* the reward

Two audiences use that checker:
- an **agent**, as an acceptance gate ("is my code good enough to ship?"), and
- an **RL trainer**, as a **reward** ("was that sample good? reward it so the model does
  more of that").

They are the same signal. So the design of the checker is not a detail — it is the
specification of what the model will *become*. If the checker only measures correctness,
that is what the agent ships and what the trained model optimizes.

## 4. Why correctness alone is the wrong reward — the two failure modes

- **Saturated.** We give an agent the strongest correctness checker we can build (a
  *differential verifier* that runs the code across thread counts {1,2,4,8} with repeats
  and compares to a trusted serial answer). For frontier models it changes **nothing** —
  they rarely write the race it's designed to catch. An independent race detector (Archer)
  agrees the accepted programs are clean. So a finer correctness reward has **no gradient
  left to give**.
- **Gameable.** Correctness is a property of *outputs*. Delete every `#pragma omp` and you
  get plain serial code that matches the reference at every thread count — a **perfect
  correctness score with zero parallelism.** A reward built on this is *maximized by the
  worst possible HPC code.* This is textbook reward hacking, and for parallel code it's
  not hypothetical — it's the reward-optimal solution.

The proof is tiny and lives in `harness/rewards.py` (run `python -m harness.rewards`):
correctness-only gives a serial program and a 6.5×-faster program the **same** score of
1.000; the two-axis reward separates them (0.125 vs 0.812).

## 5. The fix: VG-RLPT with a second gate

Your VG-RLPT framework says: treat code generation as a control problem where a
**verifier supplies the reward** and a stage of work **ends only when the verifier says
so** (execution-based termination, in the Options / temporal-abstraction sense). We keep
that skeleton and add the missing HPC piece — a **second gate**:

```
  meta-policy:   ω_correct   ──▶   ω_parallel
                 (correctness gate)   (performance gate)     ← this paper's addition

  ω_correct :  generate → differential-verify across threads → repair until robust.
               β fires when the program is correct at EVERY thread count.
  ω_parallel:  take the robust program → propose an optimization → RE-verify it is
               STILL correct → measure speedup → repair until it scales.
               β fires when 8-thread self-speedup ≥ target AND still correct.
```

The elegant part: the performance gate is **guarded by correctness** — we only ever ask
"is it fast?" of code that is already correct, and every optimization must *stay*
correct. So chasing speed can never smuggle in a wrong answer. That guard is why the
two-axis reward is safe to optimize. Code: `harness/vg_agent.py` (agent),
`harness/perf_gate.py` (the performance gate), `harness/rewards.py` (the reward).

## 6. GRPO / RLVR: how the reward trains a model

GRPO (the DeepSeek RL method) samples a **group** of answers, scores each with a
verifiable reward, and nudges the model toward the higher-scoring ones. Our claim,
stated as an RL prediction: **train with the correctness-only reward and the model drifts
toward "correct but serial"; train with the two-axis reward and it doesn't.** We test this
two ways, cheap and full:
- **Best-of-N (cheap proxy, `harness/best_of_n.py`).** Sample N answers, keep the best
  under each reward. Correctness-only keeps a *random correct* one (they all tie); two-axis
  keeps the *fastest*. The gap is the speed a correctness-only objective throws away —
  reward hacking, measured, without training anything.
- **Real GRPO (`harness/grpo_train.py`).** TRL's `GRPOTrainer` with our verifiable reward
  computed by actually compiling and timing each sample. Run it once per reward and plot
  sampled speedup vs. training step: correctness-only flat/declining, two-axis rising.

---

## 7. The experiments (what each one shows)

| # | Experiment | File | Needs | Shows |
|---|---|---|---|---|
| E1 | **Reward selection on frozen data** | `harness/reward_selection.py` | nothing (runs now) | On 12 tasks with ≥2 equally-correct programs, a correctness-only reward is *indifferent* and realizes only the pool-mean speedup; two-axis realizes the max. **3.54× → 4.23× mean; worst case 0.33× → 6.82× on the histogram.** |
| E2 | **Best-of-N with live models** | `harness/best_of_n.py` | Azure/NRP + verify pod | Same effect with fresh sampling: correctness-only ≈ mean, two-axis = max, per (task, model). Captures within-model variance the frozen data can't. |
| E3 | **VG-RLPT two-gate agent** | `harness/vg_agent.py` | Azure/NRP + verify pod | The constructive result: the correctness-only agent stops at "correct but slow"; adding ω_parallel lifts the *same task* to a scaling solution, at equal correctness. |
| E4 | **Real GRPO training** | `harness/grpo_train.py` | GPU host + TRL + g++ | The training result: correctness-only reward → parallelism decays over steps; two-axis reward → speedup climbs. |

E1 is done and is the backbone figure (`results/figures/fig_reward_selection.pdf`). E2–E4
are the live experiments you run on the machine with model + cluster access.

## 8. The benchmarks (and one you should NOT use)

- **ParEval** (Nichols et al., HPDC'24) — our base. Function-completion prompts for
  parallel code, with **serial reference implementations** we use as the trusted oracle and
  **timing drivers** we use for speedup. It's the right home: parallel tasks, correctness
  *and* performance measurable. Vendored under `external/ParEval`.
- **PCEBench** (IPDPS'25) — a newer multi-dimensional benchmark that scores LLM parallel
  code on **correctness and performance** for OpenMP/MPI. It's the closest neighbor; we
  should position against it and can reuse its task taxonomy. *(Clone + compare — planned.)*
- **KernelBench / CUDA** — if we extend to GPU kernels (where the CudaPerf line lives).
- **SWE-bench / SWE-agent — do NOT use for this.** It's Python GitHub-issue resolution.
  There is **no parallelism and no speedup axis**, so it cannot measure the thing this
  paper is about. It would be effort spent on the wrong target.

## 9. What's actually new (honest positioning)

The bare idea "reward code-gen RL with performance" is **not** new — it's an active area,
including work by the ParEval group itself. We must cite and differentiate from:

- **RLPF — "Performance-Aligned LLMs for Generating Fast Code"** (the ParEval group, 2024):
  RL from performance feedback for faster code. *They already reward on performance.*
- **Online RL with real-machine (GFLOPS) rewards for HPC** (Nagoya, 2026): matmul, CPU,
  speedup-as-reward. Very close to a naive "performance gate."
- **CudaPerf** (2026): multi-turn RL with correctness + performance + structural rewards
  for CUDA — a multi-axis reward already.
- **PCEBench** (IPDPS'25): the multi-dimensional (correctness + performance) *benchmark*.

**So our defensible contribution is NOT "add performance to the reward."** It is the
**diagnosis**: single-axis verifiable rewards for parallel code are **saturated and
gameable**, a serial program is the reward-optimal exploit, and we *measure* how much
speed a correctness-only reward discards (E1/E2) and show the two-gate structure recovers
it (E3/E4). That framing — reward *hacking / specification* for HPC codegen, with a
guarded performance gate — is what we lead with, positioned explicitly against the four
works above. If a reviewer says "performance reward isn't new," the answer is "correct,
and we're not claiming it — we're showing why the *default* reward is unsafe and measuring
the cost."

---

## 10. How to run everything (in order)

```bash
# 0) one-time: get ParEval (the oracle + drivers) and Python deps
git clone https://github.com/parallelcodefoundry/ParEval external/ParEval
python -m pip install -r requirements.txt

# 1) model access (Azure Foundry, keyless) and the verify backend
az login
$env:VERIFY_BACKEND = "nautilus"      # or the Azure VM backend

# E1 — runs with NO model access, on data already in the repo:
python -m harness.reward_selection        # -> figure + results/reward_numbers.tex

# E2 — best-of-N with live models (the GRPO proxy):
python -m harness.best_of_n --models gpt-5.4 --n 8 --types histogram reduce search

# E3 — the VG-RLPT two-gate agent:
python -m harness.vg_agent --models gpt-5.4 --types histogram reduce search --target 2.0

# E4 — real GRPO, one run per reward, on a GPU host:
python -m harness.grpo_train --reward correctness_only --types histogram reduce search
python -m harness.grpo_train --reward two_axis        --types histogram reduce search

# build the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 11. File map (what you're looking at)

```
harness/
  rewards.py           the three rewards; correctness-only is gameable, two-axis is the fix
  perf_gate.py         the performance verification gate (ω_parallel's β) — reuses timing
  vg_agent.py          the VG-RLPT two-gate agent (ω_correct → ω_parallel)
  best_of_n.py         best-of-N reward selection with live models (GRPO proxy)
  grpo_train.py        real GRPO/RLVR training with the two-axis verifiable reward
  reward_selection.py  E1: the frozen-data result + the backbone figure
  run_pareval_batch.py the original study (correctness gate + tool ablation) — reused
  measure_scaling.py   the timing pass we reuse for the performance axis
paper/                 the write-up (abstract + intro already reframed to this story)
docs/STORY.md          this file
results/               jsonl outputs + LaTeX number macros + figures
```

## 12. What you do next (the debugging loop)

1. `az login` on the machine with model access; set `VERIFY_BACKEND`.
2. Run **E2** first (fast, cheap) to confirm the model client + verify pod + perf gate all
   talk to each other end-to-end. Fix any path/name issues (model ids in `models.yaml`,
   pod name/namespace envs) there — everything else reuses the same plumbing.
3. Run **E3** on a couple of race-prone tasks; read a `logs/.../vg__*.json` transcript to
   watch ω_correct then ω_parallel in action.
4. Run **E4** on a GPU host once per reward; plot `results/grpo_*.jsonl`.
5. Ping me with any traceback and I'll fix the code — the modules are written to your
   harness's real interfaces, but only you can run them against live models.
