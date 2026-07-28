"""Batched agentic runner over ParEval OpenMP benchmarks -- fast enough for a real
study. The bottleneck is the ~30-60s fixed overhead of each `az vm run-command`.
We amortize it by VERIFYING MANY CANDIDATES PER VM CALL and by advancing every
agent's repair loop in lockstep ROUNDS: generate round-k for all still-failing
cells locally (cheap), then verify them ALL in ONE VM call.

A "cell" is one (task, model, condition, trial). Conditions:
  single_shot         -- one generation, scored once
  execute_once        -- verify at 1 thread only (single-config testing; blind to
                         races that need >1 thread); repair on failure
  differential_verify -- verify across the thread sweep vs the serial baseline;
                         repair on failure

Final robustness for EVERY cell is an independent full-sweep score, also batched.
All generations, batched verifications, and transcripts are logged (SC26 AD/AE).
"""
from __future__ import annotations
import argparse, base64, json, os, re, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

from harness import azure_llm
from harness import nrp_llm

AZ = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
RG, VM = "dsp-dev-rg", "sc26hpc"
# Verification backend: "vm" (az vm run-command, ~10 min/call) or "nautilus"
# (kubectl exec into a provisioned gcc pod, ~10 s/call). Select with VERIFY_BACKEND.
BACKEND = os.environ.get("VERIFY_BACKEND", "vm").lower()
KUBECTL = os.environ.get("KUBECTL", "kubectl")
POD = os.environ.get("VERIFY_POD", "sc26-omp-verify")
NS = os.environ.get("VERIFY_NS", "gp-engine-mizzou-radiant")
ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "external" / "ParEval" / "prompts" / "generation-prompts.json"
LOG = ROOT / "logs" / "pareval"
GEN_DIR, VER_DIR, TR_DIR = LOG / "generations", LOG / "verify", LOG / "transcripts"
for d in (GEN_DIR, VER_DIR, TR_DIR):
    d.mkdir(parents=True, exist_ok=True)
RESULTS = Path(os.environ.get("RESULTS_FILE", str(ROOT / "results" / "pareval_runs.jsonl")))

THREADS = [1, 2, 4, 8]
REPS = 2
SCORE_REPS = 2
VAL_ATTEMPTS = 8
PROBLEM_SIZE = "(1<<12)"     # small: validate() uses its own test size; init() stays cheap

SYS = ("You are an expert HPC engineer. Implement the requested function using OpenMP "
       "so it is correct under EVERY thread count. Return the COMPLETE, self-contained "
       "function definition, restating the exact signature and including any #include "
       "directives it needs, in a single ```cpp code block. Do not include a main(). "
       "Never write a shared scalar or array element from multiple threads without "
       "reduction, atomic, or critical.")
_COMMON_INC = ("#include <omp.h>\n#include <vector>\n#include <array>\n#include <cstddef>\n"
               "#include <cstdint>\n#include <string>\n#include <algorithm>\n#include <cmath>\n")


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _extract_code(t):
    m = re.findall(r"```(?:cpp|c\+\+|c)?\s*(.*?)```", t, re.DOTALL)
    return (m[-1].strip() if m else t.strip())


def _load_tasks():
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    return {p["name"]: {"prompt": p["prompt"], "type": p["problem_type"]}
            for p in data if p["parallelism_model"] == "omp"}


class Cell:
    def __init__(self, task_name, task, model, cond, trial):
        self.name, self.task, self.model = task_name, task, model
        self.cond, self.trial = cond, trial
        self.id = f"{task_name}__{model}__{cond}__t{trial}".replace(".", "-")
        self.code = ""
        self.messages = [{"role": "system", "content": SYS},
                         {"role": "user", "content": task["prompt"]}]
        self.steps = []
        self.round = 0
        self.settled = False      # loop finished (passed or budget exhausted)
        self.repaired = False
        self.final_verdict = None
        self.sweep = ""
        self.robust = False


def _llm_for(model: str):
    """Route a model id to its provider client. NRP open-weights models may be
    written bare (e.g. 'qwen3-small') or prefixed 'nrp:'; everything else goes to
    Azure. Both clients expose chat() + last_meta() with identical semantics."""
    name = model[4:] if model.startswith("nrp:") else model
    if model.startswith("nrp:") or name in set(nrp_llm.KNOWN_MODELS):
        return nrp_llm, name
    return azure_llm, name


def gen(cell: Cell):
    client, model_id = _llm_for(cell.model)
    budget = 6000 if client is nrp_llm else 2000   # reasoning models need headroom
    raw = client.chat(model_id, cell.messages, max_tokens=budget)
    meta = client.last_meta()             # LLM-emitted envelope for THIS call
    cell.code = _extract_code(raw)
    cell.messages.append({"role": "assistant", "content": f"```cpp\n{cell.code}\n```"})
    (GEN_DIR / f"{cell.id}__step{cell.round}.json").write_text(json.dumps(
        {"ts": _ts(), "model": cell.model, "api_response": meta,
         "messages": cell.messages[:-1],
         "raw_response": raw, "extracted_code": cell.code}, indent=2), encoding="utf-8")


def _cell_block(cell: Cell, threads, reps) -> str:
    gc = _COMMON_INC + cell.code
    b64 = base64.b64encode(gc.encode()).decode()
    thr = ",".join(map(str, threads))
    exp = len(threads) * reps          # runs the sweep-driver must complete
    bench = f"/opt/pareval/benchmarks/{cell.task['type']}/{cell.name}"
    cid = cell.id
    # Emit ONLY a single compact verdict line per cell (plus a short DIAG on
    # failure). Per-run detail is written to a file on the VM but NOT echoed,
    # because `az vm run-command` truncates its returned message to ~4KB, which
    # silently drops later cells' verdicts in a large batch. A correct task runs
    # in milliseconds; the 20s cap turns a deadlock/livelock into a truncated
    # (incomplete) sweep, which the run-count check below scores NOT_ROBUST.
    return f"""
D=/tmp/{cid}; rm -rf $D; mkdir -p $D; cd $D
cp {bench}/cpu.cc {bench}/baseline.hpp . 2>/dev/null
echo "{b64}" | base64 -d > generated-code.hpp
if ! g++ -std=c++17 -O2 -fopenmp -DUSE_OMP -DMAX_VALIDATION_ATTEMPTS={VAL_ATTEMPTS} \
     -DDRIVER_PROBLEM_SIZE="{PROBLEM_SIZE}" /opt/pareval/models/sweep-driver.cc cpu.cc \
     -I$D -I{bench} -I/opt/pareval -o a.out 2>cerr; then
  echo "CELL {cid} VERDICT=CRASH DIAG=$(grep -m1 error: cerr | head -c 120 | tr '\\n' ' ')"
else
  timeout 20 ./a.out {thr} {reps} 2>/dev/null > runs.txt
  npass=$(grep -c 'valid=PASS' runs.txt); nfail=$(grep -c 'valid=FAIL' runs.txt)
  sw=""
  for t in $(echo {thr} | tr , ' '); do
    tp=$(grep -c "t=$t .*valid=PASS" runs.txt); tf=$(grep -c "t=$t .*valid=FAIL" runs.txt)
    sw="$sw t$t=$tp/$((tp+tf))"
  done
  echo "CELL {cid} SWEEP$sw"
  if [ "$nfail" -gt 0 ]; then
    echo "CELL {cid} VERDICT=NOT_ROBUST DIAG=validation FAIL on $nfail of $((npass+nfail)) thread/rep runs (race/order-dependence)"
  elif [ "$npass" -lt {exp} ]; then
    echo "CELL {cid} VERDICT=NOT_ROBUST DIAG=incomplete sweep: only $npass of {exp} runs finished (deadlock/hang/crash at higher thread count)"
  else echo "CELL {cid} VERDICT=ROBUST_CORRECT"; fi
fi"""


def _invoke_vm(script_path: str, timeout: int = 3600) -> str:
    """Invoke a VM run-command, waiting out 'Conflict' (VM busy with a prior
    run-command) so batched/detached runs are robust."""
    for attempt in range(30):
        out = subprocess.run([AZ, "vm", "run-command", "invoke", "-g", RG, "-n", VM,
                              "--command-id", "RunShellScript", "--scripts", f"@{script_path}",
                              "--query", "value[0].message", "-o", "tsv"],
                             capture_output=True, text=True, timeout=timeout)
        msg = out.stdout + "\n" + out.stderr
        if "Run command extension execution is in progress" in msg or \
           ("Conflict" in msg and "ALLDONE" not in msg):
            time.sleep(20)
            continue
        return msg
    return msg


def _invoke_pod(script_path: str, timeout: int = 600) -> str:
    """Run a verification script inside the Nautilus gcc pod via `kubectl exec`.
    Streams the script as raw bytes over stdin (avoids Windows UTF-16 pipe
    corruption) and returns combined stdout+stderr. No 4KB output cap, unlike
    az vm run-command, and ~10s round-trip instead of ~10 min."""
    script = Path(script_path).read_bytes()
    for attempt in range(5):
        out = subprocess.run([KUBECTL, "exec", "-i", POD, "-n", NS, "--",
                              "bash", "-s"],
                             input=script, capture_output=True, timeout=timeout)
        msg = out.stdout.decode("utf-8", "replace") + "\n" + out.stderr.decode("utf-8", "replace")
        if "ALLDONE" in msg or "CELL " in msg:
            return msg
        if attempt < 4:
            time.sleep(5)   # transient: pod cold, exec race, etc.
            continue
        return msg
    return msg


def _invoke(script_path: str, timeout: int = 3600) -> str:
    """Dispatch a verification script to the active backend."""
    if BACKEND == "nautilus":
        return _invoke_pod(script_path, timeout=min(timeout, 600))
    return _invoke_vm(script_path, timeout=timeout)


def batch_verify(cells, threads, reps, tag) -> dict:
    """Compile+sweep cells and return {cell_id: {verdict, diag}}. Cells are split
    into small chunks (one backend call each) so the ~4KB run-command output limit
    never truncates a verdict (the nautilus backend has no such limit)."""
    if not cells:
        return {}
    CHUNK = 12
    res, raw_all = {}, []
    for ci in range(0, len(cells), CHUNK):
        chunk = cells[ci:ci + CHUNK]
        script = "#!/bin/bash\n" + "\n".join(_cell_block(c, threads, reps) for c in chunk) + '\necho ALLDONE\n'
        p = tempfile.mktemp(suffix=".sh")
        Path(p).write_text(script, newline="\n", encoding="utf-8")
        try:
            msg = _invoke(p)
        finally:
            os.unlink(p)
        raw_all.append(msg)
        for cid, verdict in re.findall(r"CELL (\S+) VERDICT=(\w+)", msg):
            res.setdefault(cid, {})["verdict"] = verdict
        for cid, d in re.findall(r"CELL (\S+) VERDICT=\w+ DIAG=(.*)", msg):
            res.setdefault(cid, {})["diag"] = d.strip()
        for cid, sw in re.findall(r"CELL (\S+) SWEEP (.*)", msg):
            res.setdefault(cid, {})["sweep"] = sw.strip()
    (VER_DIR / f"BATCH__{tag}.json").write_text(json.dumps(
        {"ts": _ts(), "threads": threads, "reps": reps, "n_cells": len(cells),
         "n_chunks": (len(cells) + CHUNK - 1) // CHUNK,
         "cell_ids": [c.id for c in cells], "raw": "\n".join(raw_all)}, indent=2), encoding="utf-8")
    return res


def main():
    tasks = _load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-5.4", "gpt-5.2"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+",
                    default=["single_shot", "execute_once", "differential_verify"])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--budget", type=int, default=2)
    args = ap.parse_args()

    names = args.tasks or [n for n, t in tasks.items()
                           if (not args.types or t["type"] in args.types)]
    done = set()
    if RESULTS.exists():
        for ln in RESULTS.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["task"], r["model"], r["condition"], r["trial"]))

    cells = [Cell(n, tasks[n], m, c, tr)
             for n in names for m in args.models for c in args.conditions
             for tr in range(args.trials) if (n, m, c, tr) not in done]
    print(f"{len(cells)} cells to run (skipped {len(done)} done)")
    if not cells:
        return

    # ---- round 0: generate all locally ----
    print("generating round 0 for all cells ...")
    for c in cells:
        gen(c)

    # single_shot cells settle immediately (no loop)
    for c in cells:
        if c.cond == "single_shot":
            c.settled = True

    # ---- agentic repair rounds (batched per round) ----
    for rnd in range(args.budget + 1):
        active = [c for c in cells if not c.settled and c.round == rnd]
        if not active:
            continue
        # verify at the condition's config
        eo = [c for c in active if c.cond == "execute_once"]
        dv = [c for c in active if c.cond == "differential_verify"]
        print(f"round {rnd}: verifying {len(eo)} execute_once + {len(dv)} differential_verify ...")
        res = {}
        if eo:
            res.update(batch_verify(eo, [1], 1, f"r{rnd}_eo"))
        if dv:
            res.update(batch_verify(dv, THREADS, REPS, f"r{rnd}_dv"))
        for c in active:
            r = res.get(c.id, {"verdict": "PARSE_ERR"})
            # execute_once verifies at 1 thread only; ROBUST_CORRECT there means
            # it passed the single-configuration check (blind to multi-thread races).
            passed = (r["verdict"] == "ROBUST_CORRECT")
            c.steps.append({"round": rnd, "verdict": r.get("verdict"), "diag": r.get("diag", "")})
            if passed or rnd == args.budget:
                c.settled = True
            else:
                c.repaired = True
                c.round += 1
                c.messages.append({"role": "user", "content":
                                   f"Your completion failed verification: {r.get('verdict')}. "
                                   f"{r.get('diag','output depends on thread scheduling (race/order-dependence)')}. "
                                   "Return the corrected complete function in one ```cpp block."})
                gen(c)
        # loop continues; next round picks up c.round==rnd+1

    # ---- independent full-sweep final scoring (batched) ----
    print(f"final scoring {len(cells)} cells (full sweep) ...")
    score = batch_verify(cells, THREADS, SCORE_REPS, "SCORE")
    for c in cells:
        r = score.get(c.id, {"verdict": "PARSE_ERR"})
        c.final_verdict = r.get("verdict")
        c.sweep = r.get("sweep", "")
        c.robust = (c.final_verdict == "ROBUST_CORRECT")

    # ---- persist transcripts + result rows ----
    with RESULTS.open("a", encoding="utf-8") as fh:
        for c in cells:
            (TR_DIR / f"{c.id}.json").write_text(json.dumps(
                {"ts": _ts(), "task": c.name, "type": c.task["type"], "model": c.model,
                 "condition": c.cond, "trial": c.trial, "steps": c.steps,
                 "messages": c.messages, "final_code": c.code,
                 "final_verdict": c.final_verdict, "sweep": c.sweep,
                 "robust": c.robust}, indent=2), encoding="utf-8")
            fh.write(json.dumps(
                {"ts": _ts(), "task": c.name, "type": c.task["type"], "model": c.model,
                 "condition": c.cond, "trial": c.trial, "verdict": c.final_verdict,
                 "robust": c.robust, "sweep": c.sweep,
                 "n_iters": len(c.steps), "repaired": c.repaired}) + "\n")
    n_rob = sum(c.robust for c in cells)
    print(f"\ndone: {n_rob}/{len(cells)} robust. wrote {RESULTS}")


if __name__ == "__main__":
    main()
