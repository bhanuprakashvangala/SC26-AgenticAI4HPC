"""GRPO / RLVR post-training with a TWO-AXIS verifiable reward.

This is the training end of the argument: if a correctness-only reward is gameable,
then post-training a model with GRPO on it should drift toward "correct but not
parallel", while post-training on the two-axis reward should not. This script lets you
run both and compare.

It uses TRL's GRPOTrainer (the DeepSeek-style Group Relative Policy Optimization) with
a *verifiable* reward computed by actually compiling and running each sampled
completion -- the RLVR setting. Two reward functions are provided, selectable with
--reward:

  correctness_only : rewards.correctness_only(shaped=True)   -- the gameable objective
  two_axis         : rewards.two_axis(shaped=True)           -- the fixed objective

The reward runs LOCALLY (g++ -fopenmp on the training host) rather than over the
verification pod, because GRPO needs thousands of fast reward evaluations. It reuses
ParEval's serial reference (correctness) and timing driver (speedup) from the vendored
external/ParEval checkout, so the training reward is the same quantity the study
measures -- just computed in-process for speed.

REQUIREMENTS (run on a GPU host, not this laptop):
  pip install "trl>=0.12" transformers accelerate peft datasets torch
  a C++ toolchain with OpenMP (g++), and external/ParEval present (see README step 0).

Usage:
  python -m harness.grpo_train --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --reward two_axis --types histogram reduce search scan --steps 500
  # run again with --reward correctness_only to get the comparison arm.

The headline plot the paper wants: mean 8-thread self-speedup of the policy's samples
vs. training step, for the two reward arms -- correctness-only stays flat/declines,
two-axis rises. Sampled speedups are logged to results/grpo_<reward>.jsonl each eval.
"""
from __future__ import annotations
import argparse, base64, json, os, re, subprocess, tempfile
from pathlib import Path

from harness import rewards

ROOT = Path(__file__).resolve().parents[1]
PAREVAL = ROOT / "external" / "ParEval"
PROMPTS = PAREVAL / "prompts" / "generation-prompts.json"
DRIVERS = PAREVAL / "drivers" / "cpp"          # omp-driver.cc / sweep-driver.cc live here
BENCH = PAREVAL / "benchmarks"
GXX = os.environ.get("CXX", "g++")
THREADS = [1, 2, 4, 8]
_INC = ("#include <omp.h>\n#include <vector>\n#include <array>\n#include <cstddef>\n"
        "#include <cstdint>\n#include <string>\n#include <algorithm>\n#include <cmath>\n")
_CODE_RE = re.compile(r"```(?:cpp|c\+\+|c)?\s*(.*?)```", re.DOTALL)


def _extract(text: str) -> str:
    m = _CODE_RE.findall(text)
    return (m[-1].strip() if m else text.strip())


# --------------------------- local verifiable reward ---------------------------
def local_verify(code: str, task: str, ptype: str, size: str = "(1<<22)") -> dict:
    """Compile `code` for `task`, run the correctness sweep and a quick timing pass
    locally. Returns {verdict, sweep, speedup8}. Best-effort: any failure -> CRASH.

    Correctness uses vm/sweep-driver.cc semantics (validate across {1,2,4,8} x reps);
    timing uses ParEval's omp-driver. Both drivers are expected under external/ParEval.
    """
    bench = BENCH / ptype / task
    sweep_driver = ROOT / "vm" / "sweep-driver.cc"
    omp_driver = DRIVERS / "omp-driver.cc"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "generated-code.hpp").write_text(_INC + code, encoding="utf-8")
        for f in ("cpu.cc", "baseline.hpp"):
            src = bench / f
            if src.exists():
                (d / f).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        # ---- correctness sweep ----
        exe = d / "a.out"
        cc = subprocess.run(
            [GXX, "-std=c++17", "-O2", "-fopenmp", "-DUSE_OMP", "-DMAX_VALIDATION_ATTEMPTS=8",
             f"-DDRIVER_PROBLEM_SIZE={size}", str(sweep_driver), str(d / "cpu.cc"),
             f"-I{d}", f"-I{bench}", f"-I{PAREVAL}", "-o", str(exe)],
            capture_output=True, text=True)
        if cc.returncode != 0:
            return {"verdict": "CRASH", "sweep": "", "speedup8": None}
        run = subprocess.run([str(exe), ",".join(map(str, THREADS)), "2"],
                             capture_output=True, text=True, timeout=60)
        lines = run.stdout
        sweep = " ".join(
            f"t{t}={lines.count(f't={t} ')//1 and sum(1 for _ in re.findall(f't={t} .*valid=PASS', lines))}"
            f"/{sum(1 for _ in re.findall(f't={t} ', lines))}" for t in THREADS)
        nfail = len(re.findall(r"valid=FAIL", lines))
        npass = len(re.findall(r"valid=PASS", lines))
        verdict = ("ROBUST_CORRECT" if nfail == 0 and npass >= len(THREADS) * 2
                   else "NOT_ROBUST")
        # ---- quick timing (only if correct) ----
        speedup = None
        if verdict == "ROBUST_CORRECT" and omp_driver.exists():
            texe = d / "s.out"
            tc = subprocess.run(
                [GXX, "-std=c++17", "-O3", "-march=native", "-fopenmp", "-DUSE_OMP",
                 "-DMAX_VALIDATION_ATTEMPTS=3", f"-DDRIVER_PROBLEM_SIZE={size}",
                 str(omp_driver), str(d / "cpu.cc"), f"-I{d}", f"-I{bench}", f"-I{PAREVAL}",
                 "-o", str(texe)], capture_output=True, text=True)
            if tc.returncode == 0:
                times = {}
                for t in THREADS:
                    best = None
                    for _ in range(3):
                        r = subprocess.run([str(texe), str(t)], capture_output=True,
                                           text=True, timeout=60,
                                           env={**os.environ, "OMP_NUM_THREADS": str(t)})
                        mt = re.search(r"^Time:\s*([\d.eE+-]+)", r.stdout, re.M)
                        if mt:
                            v = float(mt.group(1))
                            best = v if best is None else min(best, v)
                    if best:
                        times[t] = best
                if times.get(1) and times.get(8):
                    speedup = times[1] / times[8]
        return {"verdict": verdict, "sweep": sweep, "speedup8": speedup}


def make_reward_fn(kind: str, log_path: Path):
    """Build a TRL reward function of signature (prompts, completions, **cols)->list."""
    def reward_fn(prompts, completions, **cols):
        tasks = cols.get("task"); ptypes = cols.get("ptype")
        out = []
        rec = []
        for i, comp in enumerate(completions):
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            v = local_verify(_extract(text), tasks[i], ptypes[i])
            if kind == "correctness_only":
                r = rewards.correctness_only(v["verdict"], v["sweep"], shaped=True)
            else:
                r = rewards.two_axis(v["verdict"], v["speedup8"], v["sweep"], shaped=True)
            out.append(float(r))
            rec.append({"task": tasks[i], **v, "reward": r})
        with log_path.open("a", encoding="utf-8") as fh:
            for x in rec:
                fh.write(json.dumps(x) + "\n")
        return out
    reward_fn.__name__ = f"reward_{kind}"
    return reward_fn


def build_dataset(types):
    from datasets import Dataset
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    rows = [{"prompt": p["prompt"], "task": p["name"], "ptype": p["problem_type"]}
            for p in data if p["parallelism_model"] == "omp"
            and (not types or p["problem_type"] in types)]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--reward", choices=["two_axis", "correctness_only"], default="two_axis")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group-size", type=int, default=8, help="GRPO samples per prompt")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from trl import GRPOConfig, GRPOTrainer          # imported here so --help needs no GPU deps

    out_dir = args.out or str(ROOT / "runs" / f"grpo_{args.reward}")
    log_path = ROOT / "results" / f"grpo_{args.reward}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(args.types)
    print(f"GRPO ({args.reward}) on {len(ds)} prompts, group size {args.group_size}, "
          f"{args.steps} steps -> {out_dir}")

    cfg = GRPOConfig(
        output_dir=out_dir,
        learning_rate=args.lr,
        num_generations=args.group_size,
        max_steps=args.steps,
        per_device_train_batch_size=args.group_size,
        logging_steps=5,
        save_steps=100,
        bf16=True,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=make_reward_fn(args.reward, log_path),
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"done. reward trace -> {log_path}")


if __name__ == "__main__":
    main()
