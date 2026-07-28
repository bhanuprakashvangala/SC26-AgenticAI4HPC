"""Measure PARALLEL SCALING of the programs that scored configuration-robust.

The differential verifier establishes an axis of *correctness*. This module adds
the second axis the reviewer demands: are the accepted programs actually parallel,
or did the models buy race-freedom with over-synchronization (critical/atomic where
a reduction belongs, serialized merges) that is robust yet does not scale?

For each robust program we compile the model's code against ParEval's timing driver
(omp-driver.cc: validate once, then time NITER runs of compute() at a given thread
count) with a per-task problem size large enough that a genuinely parallel kernel
scales. We run threads in {1,2,4,8}, take the min wall-time over reps (noise floor),
and report:
  self-speedup  S(p) = T(1)/T(p)     -- does adding threads help THIS program?
  parallel-eff  E(p) = S(p)/p         -- 1.0 = perfect, ~1/p = serial-in-disguise
  vs-serial     BestSequential/T(p)   -- is it even faster than the serial ref?
A program that is robust but has S(8) ~ 1 is "correct but not parallel".

Runs on the Nautilus pod (VERIFY_BACKEND=nautilus) via kubectl exec.
Usage: python -m harness.measure_scaling [--condition differential_verify]
Output: results/scaling.jsonl
"""
from __future__ import annotations
import argparse, json, re, subprocess, tempfile, base64, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "logs" / "pareval" / "transcripts"
OUT = ROOT / "results" / "scaling.jsonl"
POD = os.environ.get("VERIFY_POD", "sc26-omp-verify")
NS = os.environ.get("VERIFY_NS", "gp-engine-mizzou-radiant")
KUBECTL = os.environ.get("KUBECTL", "kubectl")

THREADS = [1, 2, 4, 8]
REPS = 3                      # runs per (program, thread count); we take the min
VAL_ATTEMPTS = 3

# Per-task problem size (the DRIVER_PROBLEM_SIZE macro). O(n) kernels get a large
# vector; O(n^2) gemv gets a matrix dimension chosen for comparable total work.
SIZE_ONYX = 1 << 24          # 16.7M elements for linear-work kernels
TASK_SIZE = {
    "04_dense_la_gemv": 1 << 12,           # O(n^2): 4096 -> ~16M ops
    "17_graph_highest_degree": 1 << 12,    # adjacency n x n
}
_COMMON_INC = ("#include <omp.h>\n#include <vector>\n#include <array>\n#include <cstddef>\n"
               "#include <cstdint>\n#include <string>\n#include <algorithm>\n#include <cmath>\n")


def _size_for(task):
    return TASK_SIZE.get(task, SIZE_ONYX)


def _load_robust(condition):
    cells = []
    for f in TR.glob(f"*__{condition}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("robust") and d.get("final_code"):
            cells.append(d)
    return cells


def _bench_block(cell):
    """Bash to compile one program with the timing driver and run the thread sweep."""
    code = _COMMON_INC + cell["final_code"]
    b64 = base64.b64encode(code.encode()).decode()
    task = cell["task"]
    bench = f"/opt/pareval/benchmarks/{cell['type']}/{task}"
    size = _size_for(task)
    cid = f"{task}__{cell['model']}__{cell['condition']}__t{cell.get('trial',0)}".replace(".", "-")
    thr = " ".join(str(t) for t in THREADS)
    return f"""
D=/tmp/S_{cid}; rm -rf $D; mkdir -p $D; cd $D
cp {bench}/cpu.cc {bench}/baseline.hpp . 2>/dev/null
echo "{b64}" | base64 -d > generated-code.hpp
if ! g++ -std=c++17 -O3 -march=native -fopenmp -DUSE_OMP \
     -DMAX_VALIDATION_ATTEMPTS={VAL_ATTEMPTS} -DDRIVER_PROBLEM_SIZE="{size}" \
     /opt/pareval/models/omp-driver.cc cpu.cc -I$D -I{bench} -I/opt/pareval \
     -o s.out 2>cerr; then
  echo "SCELL {cid} COMPILE_FAIL"
else
  for t in {thr}; do
    for r in 1 2 3; do
      OMP_NUM_THREADS=$t timeout 60 ./s.out $t 2>/dev/null \
        | awk -v T=$t '/^Time:/{{print "SCELL {cid} t="T" time="$2}} /^BestSequential:/{{print "SCELL {cid} t="T" best="$2}} /^Validation:/{{print "SCELL {cid} val="$2}}'
    done
  done
fi"""


def _invoke_pod(script: str, timeout=900) -> str:
    out = subprocess.run([KUBECTL, "exec", "-i", POD, "-n", NS, "--", "bash", "-s"],
                         input=script.encode(), capture_output=True, timeout=timeout)
    return out.stdout.decode("utf-8", "replace") + "\n" + out.stderr.decode("utf-8", "replace")


def _parse(msg):
    """cid -> {threads:{t:min_time}, best:{t:min}, compiled, valid}."""
    res = {}
    for cid in set(re.findall(r"SCELL (\S+)", msg)):
        res[cid] = {"threads": {}, "best": {}, "compiled": True, "valid": None}
    for cid in re.findall(r"SCELL (\S+) COMPILE_FAIL", msg):
        res.setdefault(cid, {"threads": {}, "best": {}})["compiled"] = False
    for cid, v in re.findall(r"SCELL (\S+) val=(\w+)", msg):
        res.setdefault(cid, {"threads": {}, "best": {}, "compiled": True})["valid"] = v
    for cid, t, tm in re.findall(r"SCELL (\S+) t=(\d+) time=([\d.eE+-]+)", msg):
        d = res.setdefault(cid, {"threads": {}, "best": {}, "compiled": True})
        t = int(t); tm = float(tm)
        d["threads"][t] = min(tm, d["threads"].get(t, float("inf")))
    for cid, t, bs in re.findall(r"SCELL (\S+) t=(\d+) best=([\d.eE+-]+)", msg):
        d = res.setdefault(cid, {"threads": {}, "best": {}, "compiled": True})
        t = int(t); bs = float(bs)
        d["best"][t] = min(bs, d["best"].get(t, float("inf")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="differential_verify")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cells = _load_robust(args.condition)
    if args.limit:
        cells = cells[:args.limit]
    print(f"{len(cells)} robust {args.condition} programs to time")

    OUT.write_text("", encoding="utf-8")
    # one program per exec keeps timing clean (no cross-program CPU contention)
    for i, cell in enumerate(cells, 1):
        script = "#!/bin/bash\n" + _bench_block(cell) + "\necho ALLDONE\n"
        try:
            msg = _invoke_pod(script)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(cells)}] {cell['task']} EXEC_ERR {e}")
            continue
        res = _parse(msg)
        cid = f"{cell['task']}__{cell['model']}__{cell['condition']}__t{cell.get('trial',0)}".replace(".", "-")
        r = res.get(cid, {})
        th = r.get("threads", {})
        best = r.get("best", {})
        s8 = (th.get(1) / th.get(8)) if th.get(1) and th.get(8) else None
        vs_serial8 = (best.get(8) / th.get(8)) if best.get(8) and th.get(8) else None
        row = {"task": cell["task"], "type": cell["type"], "model": cell["model"],
               "condition": cell["condition"], "compiled": r.get("compiled", False),
               "valid": r.get("valid"), "times": th, "best": best,
               "self_speedup8": s8, "vs_serial8": vs_serial8,
               "size": _size_for(cell["task"])}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        tag = f"S8={s8:.2f}" if s8 else ("COMPILE_FAIL" if not r.get("compiled") else "no-time")
        print(f"  [{i}/{len(cells)}] {cell['task']:32s} {cell['model']:12s} {tag}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
