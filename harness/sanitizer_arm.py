"""Sanitizer arm: cross-check the output-differential verdict against a
happens-before race detector (LLVM Archer = ThreadSanitizer + OpenMP OMPT
annotations). Answers the reviewer's first question: is our '0 silent races'
real, or did output-differential testing at tau<=8 simply lack the power to see
races that Archer would flag?

For each program we compile the model's code + the benchmark's cpu.cc against
ParEval's omp-driver with clang -fopenmp -fsanitize=thread, then run under
Archer (LD_PRELOAD libarcher.so) with ASLR disabled (setarch -R, required for
TSan's shadow memory). We classify a program RACY only if a TSan 'data race'
report has a stack frame in USER code (the generated function or cpu.cc), not in
the libomp/pthread runtime (which produces benign, archer-residual reports).

Runs on the Nautilus pod. Usage:
  python -m harness.sanitizer_arm --condition differential_verify [--robust-only]
Output: results/sanitizer.jsonl
"""
from __future__ import annotations
import argparse, json, re, subprocess, base64, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "logs" / "pareval" / "transcripts"
OUT = ROOT / "results" / "sanitizer.jsonl"
POD = os.environ.get("VERIFY_POD", "sc26-omp-verify")
NS = os.environ.get("VERIFY_NS", "gp-engine-mizzou-radiant")
KUBECTL = os.environ.get("KUBECTL", "kubectl")
ARCHER = "/usr/lib/llvm-14/lib/libarcher.so"
THREADS = 4
RUNS = 3      # TSan is nondeterministic; several runs raise detection power
VAL_ATTEMPTS = 3
PROBLEM_SIZE = "(1<<16)"     # big enough to interleave, small enough for TSan overhead
_COMMON_INC = ("#include <omp.h>\n#include <vector>\n#include <array>\n#include <cstddef>\n"
               "#include <cstdint>\n#include <string>\n#include <algorithm>\n#include <cmath>\n")


def _load(condition, robust_only):
    cells = []
    for f in TR.glob(f"*__{condition}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("final_code"):
            continue
        if robust_only and not d.get("robust"):
            continue
        cells.append(d)
    return cells


def _block(cell):
    code = _COMMON_INC + cell["final_code"]
    b64 = base64.b64encode(code.encode()).decode()
    task = cell["task"]
    bench = f"/opt/pareval/benchmarks/{cell['type']}/{task}"
    cid = f"{task}__{cell['model']}__{cell['condition']}__t{cell.get('trial',0)}".replace(".", "-")
    # Compile with clang + TSan + OpenMP; run under Archer with ASLR off.
    # Emit, per run, whether TSan saw a data race whose report cites USER code.
    return f"""
D=/tmp/TS_{cid}; rm -rf $D; mkdir -p $D; cd $D
cp {bench}/cpu.cc {bench}/baseline.hpp . 2>/dev/null
echo "{b64}" | base64 -d > generated-code.hpp
if ! clang++ -std=c++17 -O1 -g -fopenmp -fsanitize=thread -DUSE_OMP \
     -DMAX_VALIDATION_ATTEMPTS={VAL_ATTEMPTS} -DDRIVER_PROBLEM_SIZE="{PROBLEM_SIZE}" \
     /opt/pareval/models/omp-driver.cc cpu.cc -I$D -I{bench} -I/opt/pareval \
     -o t.out 2>cerr; then
  echo "TCELL {cid} COMPILE_FAIL"
else
  user=0; any=0
  for r in 1 2 3; do
    OMP_NUM_THREADS={THREADS} setarch $(uname -m) -R env LD_PRELOAD={ARCHER} \
      ./t.out {THREADS} 2>ts.txt >/dev/null || true
    if grep -q "data race" ts.txt; then any=1; fi
    # a USER-code race: a stack frame symbolized to the model's generated code.
    # libomp/pthread-only reports are Archer runtime residue and are NOT counted.
    if grep -A30 "data race" ts.txt | grep -q "generated-code.hpp"; then
      user=1
    fi
  done
  echo "TCELL {cid} tsan_any=$any user_race=$user"
fi"""


def _invoke(script, timeout=1200):
    out = subprocess.run([KUBECTL, "exec", "-i", POD, "-n", NS, "--", "bash", "-s"],
                         input=script.encode(), capture_output=True, timeout=timeout)
    return out.stdout.decode("utf-8", "replace") + "\n" + out.stderr.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="differential_verify")
    ap.add_argument("--robust-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cells = _load(args.condition, args.robust_only)
    if args.limit:
        cells = cells[:args.limit]
    print(f"{len(cells)} programs to sanitize (condition={args.condition}, "
          f"robust_only={args.robust_only})")
    OUT.write_text("", encoding="utf-8")

    n_user = n_compile_fail = n_ok = 0
    for i, cell in enumerate(cells, 1):
        script = "#!/bin/bash\n" + _block(cell) + "\necho ALLDONE\n"
        try:
            msg = _invoke(script)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(cells)}] {cell['task']} EXEC_ERR {e}")
            continue
        cid = f"{cell['task']}__{cell['model']}__{cell['condition']}__t{cell.get('trial',0)}".replace(".", "-")
        cf = f"TCELL {cid} COMPILE_FAIL" in msg
        m = re.search(rf"TCELL {re.escape(cid)} tsan_any=(\d) user_race=(\d)", msg)
        user_race = bool(m and m.group(2) == "1")
        robust = cell.get("robust")
        row = {"task": cell["task"], "type": cell["type"], "model": cell["model"],
               "condition": cell["condition"], "robust": robust,
               "compiled": not cf, "tsan_any": bool(m and m.group(1) == "1"),
               "user_race": user_race}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if cf:
            n_compile_fail += 1; tag = "COMPILE_FAIL(tsan)"
        elif user_race:
            n_user += 1; tag = "*** USER RACE ***"
        else:
            n_ok += 1; tag = "clean"
        flag = "  <-- robust-but-racy!" if (user_race and robust) else ""
        print(f"  [{i}/{len(cells)}] {cell['task']:30s} {cell['model']:12s} {tag}{flag}")
    print(f"\nuser-code races: {n_user}   clean: {n_ok}   tsan-compile-fail: {n_compile_fail}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
