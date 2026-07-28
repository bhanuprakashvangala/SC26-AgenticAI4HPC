"""Verify already-generated programs across the full thread sweep, without any
new LLM calls. Reads saved generation transcripts from logs/pareval/generations,
takes each cell's FINAL program (highest step), and runs the differential
verifier (batch_verify) over the thread sweep on the active backend.

Purpose: answer the race question directly on real model output -- of the
programs an agent ACCEPTED under a given condition, how many pass at one thread
but fail at higher thread counts (a silent race the single-thread check misses)?

Usage:
  $env:VERIFY_BACKEND="nautilus"
  python -m harness.verify_saved --model qwen3-small --out results/qwen_sweep.jsonl
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from harness import run_pareval_batch as R

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "logs" / "pareval" / "generations"


def _final_gen_per_cell(model_tag: str):
    """Return {(task, cond): (cell_id, code)} using the highest-step generation
    for each (task, cond) of the given model."""
    pat = re.compile(r"^(?P<task>.+?)__(?P<model>.+?)__(?P<cond>single_shot|execute_once|"
                     r"differential_verify)__t(?P<trial>\d+)__step(?P<step>\d+)\.json$")
    best: dict[tuple, tuple] = {}
    for f in GEN.glob(f"*{model_tag}*.json"):
        m = pat.match(f.name)
        if not m or m.group("model") != model_tag.replace(".", "-"):
            continue
        key = (m.group("task"), m.group("cond"), int(m.group("trial")))
        step = int(m.group("step"))
        if key not in best or step > best[key][0]:
            best[key] = (step, f)
    out = {}
    for (task, cond, trial), (_step, f) in best.items():
        d = json.loads(f.read_text(encoding="utf-8"))
        code = d.get("extracted_code", "") or R._extract_code(d.get("raw_response", ""))
        out[(task, cond, trial)] = code
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-small")
    ap.add_argument("--out", default="results/verify_saved.jsonl")
    ap.add_argument("--conditions", nargs="+",
                    default=["single_shot", "execute_once", "differential_verify"])
    args = ap.parse_args()

    tasks = R._load_tasks()
    gens = _final_gen_per_cell(args.model)
    cells = []
    for (task, cond, trial), code in sorted(gens.items()):
        if cond not in args.conditions or not code.strip():
            continue
        if task not in tasks:
            continue
        c = R.Cell(task, tasks[task], args.model, cond, trial)
        c.code = code
        cells.append(c)
    print(f"{len(cells)} saved programs to verify (model={args.model})")
    if not cells:
        return

    # Full differential sweep, scoring reps -- the authoritative robustness test.
    score = R.batch_verify(cells, R.THREADS, R.SCORE_REPS, f"VERIFYSAVED_{args.model}")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    races = semantic = compile_fail = robust = 0
    with outp.open("w", encoding="utf-8") as fh:
        for c in cells:
            r = score.get(c.id, {"verdict": "PARSE_ERR"})
            sweep = r.get("sweep", "")
            verdict = r.get("verdict", "PARSE_ERR")
            # pass@1 but fail@N => race (silent under a single-thread check)
            t1 = re.search(r"t1=(\d+)/(\d+)", sweep)
            hi = re.findall(r"t(?:2|4|8)=(\d+)/(\d+)", sweep)
            pass1 = bool(t1) and t1.group(1) == t1.group(2) and t1.group(2) != "0"
            failN = any(p != tot for p, tot in hi) if hi else False
            is_race = pass1 and failN
            if verdict == "ROBUST_CORRECT":
                robust += 1
            elif is_race:
                races += 1
            elif verdict == "CRASH":
                compile_fail += 1
            else:
                semantic += 1
            fh.write(json.dumps({
                "task": c.name, "type": c.task["type"], "model": args.model,
                "condition": c.cond, "trial": c.trial, "verdict": verdict,
                "sweep": sweep, "diag": r.get("diag", ""),
                "robust": verdict == "ROBUST_CORRECT", "race": is_race}) + "\n")
            tag = "RACE" if is_race else verdict
            print(f"  {c.name:34s} {c.cond:20s} {tag:14s} [{sweep}]")
    print(f"\nrobust={robust} race={races} semantic={semantic} compile={compile_fail}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
