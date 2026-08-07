"""
diag_core.py  --  the grounded evidence layer for the agentic HPC code diagnostician.

This is the HPC analogue of the churn project's SHAP layer: it does not let a model
*assert* anything about a program's parallelism -- it OBSERVES what was actually measured
when that program was compiled and run. Every fact the agent is allowed to say comes from
here, backed by a real measurement in the frozen run pool:

  logs/pareval/generations/*.json   the model-generated OpenMP source + the task spec
  results/scaling.jsonl             per-thread wall-times, self-speedup, efficiency (real timing)
  results/sanitizer.jsonl           LLVM Archer / TSan race verdict per program
  results/oversync.jsonl            the synchronization idiom the model chose (critical/atomic/reduction)
  results/pareval_runs.jsonl        configuration-robust correctness verdict

A "program" is one (task, model, condition) that the differential verifier accepted and that
was then timed. There are 63 of them. The agent reads a program's *whole* story -- correct at
every thread count, yet S(8)=0.41x because the reduction sits behind a whole-body critical --
and is forbidden (by the critic, elsewhere) from claiming any number this layer did not return.

Pure offline: no compiler, no VM, no network. The measurements were taken once, on the
verification backend, and frozen -- exactly as the churn agent reads a frozen feature table.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as _st
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_RESULTS = os.path.join(_ROOT, "results")
_GENS = os.path.join(_ROOT, "logs", "pareval", "generations")

_NMAX = 8  # the top of the thread sweep; the gate is min(S_n/n, 1) at n = _NMAX


# --------------------------------------------------------------- loading --------
def _load_jsonl(name: str) -> list[dict]:
    fp = os.path.join(_RESULTS, name)
    rows = []
    if os.path.exists(fp):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _thread_map(raw) -> dict:
    """scaling.jsonl stores times/best as a python-repr dict string; parse leniently."""
    if isinstance(raw, dict):
        return {int(k): _f(v) for k, v in raw.items()}
    if not isinstance(raw, str):
        return {}
    out = {}
    for k, v in re.findall(r"'(\d+)'\s*:\s*([0-9.eE+-]+)", raw):
        out[int(k)] = _f(v)
    return out


def _load_generations() -> dict:
    """Index the model-generated source by (task, condition, model), keeping the last repair step."""
    idx = {}
    for f in glob.glob(os.path.join(_GENS, "*.json")):
        parts = os.path.basename(f)[:-5].split("__")
        if len(parts) < 5:
            continue
        task, cond = parts[0], parts[2]
        step = int(re.sub(r"\D", "", parts[4]) or 0)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model = d.get("model")
        code = d.get("extracted_code") or ""
        spec = ""
        for m in d.get("messages", []):
            if m.get("role") == "user":
                spec = m.get("content", "")
        key = (task, cond, model)
        if key not in idx or step >= idx[key]["step"]:
            idx[key] = {"step": step, "code": code, "spec": spec}
    return idx


# ------------------------------------------------- serial projection (static) ---
_PRAGMA_RE = re.compile(r"^\s*#\s*pragma\s+omp\b.*$", re.MULTILINE)
# constructs whose removal can change RESULTS, not just speed -- these bound the projection honestly
_SEMANTIC_OMP = re.compile(r"\b(ordered|scan|simd\b.*\breduction|declare\s+reduction)\b")
_RUNTIME_ID = re.compile(r"\bomp_get_(thread_num|num_threads)\s*\(")


def serial_projection(code: str) -> dict:
    """Delete every OpenMP directive and predict, statically, whether the program stays correct.

    This is the executable core of the degeneracy argument: for an ordinary parallel-for (even one
    guarded by reduction / atomic / critical) the directive changes *scheduling*, not the computed
    value, so the stripped program is the serial reference and stays correct -- at the serial floor
    of the gated reward. We flag the cases where removal is NOT obviously value-preserving (ordered,
    scan, thread-id-indexed writes) rather than silently claiming them. Labelled static: a live
    re-verify would confirm, but the structure already decides the ordinary case.
    """
    pragmas = _PRAGMA_RE.findall(code or "")
    stripped = _PRAGMA_RE.sub("", code or "")
    uses_runtime_id = bool(_RUNTIME_ID.search(code or ""))
    uses_semantic = bool(_SEMANTIC_OMP.search(code or ""))
    # heuristic: if the parallelism is expressed purely through directives on an otherwise
    # sequential loop nest, stripping yields the serial program => still correct.
    preserves = not (uses_runtime_id or uses_semantic)
    return {
        "n_omp_directives": len(pragmas),
        "removed_directives": [p.strip() for p in pragmas],
        "predicted_still_correct": preserves,
        "reason": (
            "no directive changes the computed value (schedule-only); stripped program is the "
            "serial reference"
            if preserves else
            "removal may change results (thread-id indexing or ordered/scan reduction) -- would "
            "need a live re-verify to bound"
        ),
        "analysis": "static (source transform); not re-executed",
    }


# ----------------------------------------------------------------- session ------
class Session:
    """The frozen, grounded evidence pool. Read-only; every number is a real measurement."""

    def __init__(self, verbose: bool = True):
        self._gen = _load_generations()
        scaling = _load_jsonl("scaling.jsonl")
        san = {self._key(r): r for r in _load_jsonl("sanitizer.jsonl")}
        # oversync.jsonl carries no `condition` column -> join on (task, model) only
        ov = {(r.get("task"), r.get("model")): r for r in _load_jsonl("oversync.jsonl")}
        runs = {}
        for r in _load_jsonl("pareval_runs.jsonl"):
            runs[self._key(r)] = r  # last wins == final verdict

        self.programs: list[dict] = []
        for s in scaling:
            key = self._key(s)
            prog = self._assemble(s, san.get(key), ov.get((s["task"], s["model"])), runs.get(key),
                                  self._gen.get((s["task"], s["condition"], s["model"])))
            self.programs.append(prog)
        # stable order: slowest-scaling first (the interesting end)
        self.programs.sort(key=lambda p: (p["self_speedup8"] if p["self_speedup8"] is not None else 1e9))
        self._runs_all = _load_jsonl("pareval_runs.jsonl")
        if verbose:
            sp = [p["self_speedup8"] for p in self.programs if p["self_speedup8"] is not None]
            print("[session] %d accepted+timed programs | self-speedup(8) %.2fx--%.2fx median %.2fx | "
                  "%d slower-than-serial" % (len(self.programs), min(sp), max(sp), _st.median(sp),
                                             sum(x < 1.0 for x in sp)))

    @staticmethod
    def _key(r: dict) -> tuple:
        return (r.get("task"), r.get("condition"), r.get("model"))

    # ---- assembly ----
    def _assemble(self, s, san, ov, run, gen) -> dict:
        times = _thread_map(s.get("times"))
        best = _thread_map(s.get("best"))
        s8 = _f(s.get("self_speedup8"))
        eff8 = (s8 / _NMAX) if s8 is not None else None
        r_corr = 1.0 if str(s.get("valid", "")).upper() == "PASS" else 0.0
        r_gate = (r_corr * min(eff8, 1.0)) if eff8 is not None else None
        code = gen["code"] if gen else ""
        proj = serial_projection(code) if code else None
        return {
            "task": s.get("task"), "type": s.get("type"), "model": s.get("model"),
            "condition": s.get("condition"),
            "source_code": code,
            "task_spec": (gen["spec"] if gen else ""),
            "correct_robust": bool(r_corr),          # accepted by the differential verifier at every thread count
            "compiled": str(s.get("compiled", "")).lower() == "true",
            "times_by_threads": times,
            "serial_ref_times": best,
            "self_speedup8": s8,
            "parallel_efficiency8": (round(eff8, 3) if eff8 is not None else None),
            "vs_serial8": _f(s.get("vs_serial8")),
            "problem_size": _f(s.get("size")),
            "race_check": self._race(san),
            "sync_idiom": self._idiom(ov),
            "reward_correctness_only": r_corr,
            "reward_efficiency_gated": (round(r_gate, 3) if r_gate is not None else None),
            "serial_projection": proj,
            "verdict": self._verdict(r_corr, s8),
        }

    @staticmethod
    def _race(san) -> dict:
        if not san:
            return {"checked": False}
        return {"checked": True,
                "tsan_race_detected": bool(san.get("tsan_any")),
                "user_visible_race": bool(san.get("user_race")),
                "clean": not (bool(san.get("tsan_any")) or bool(san.get("user_race")))}

    @staticmethod
    def _idiom(ov) -> dict:
        if not ov:
            return {"known": False}
        return {"known": True, "primary": ov.get("primary"), "flags": ov.get("flags")}

    @staticmethod
    def _verdict(r_corr: float, s8: Optional[float]) -> str:
        if r_corr < 1.0:
            return "not accepted (failed correctness)"
        if s8 is None:
            return "correct; scaling not measured"
        if s8 < 1.0:
            return "correct but SLOWER than serial"
        if s8 < 2.0:
            return "correct but barely parallel (<2x on 8 threads)"
        if s8 < 0.5 * _NMAX:
            return "correct, partially scaling"
        return "correct and scaling"

    # ---- resolution ----
    def resolve(self, selector) -> int:
        if isinstance(selector, int):
            if 0 <= selector < len(self.programs):
                return selector
            raise ValueError("index out of range")
        s = str(selector).strip().lower()
        sp = [(i, p["self_speedup8"]) for i, p in enumerate(self.programs)
              if p["self_speedup8"] is not None]
        if s in ("slowest", "worst", "correct-but-serial", "least parallel", "worst-scaling"):
            return min(sp, key=lambda t: t[1])[0]
        if s in ("fastest", "best", "best-scaling", "most parallel"):
            return max(sp, key=lambda t: t[1])[0]
        if s in ("below-serial", "slower-than-serial"):
            cands = [i for i, v in sp if v < 1.0]
            if cands:
                return cands[0]
            raise ValueError("no below-serial program in the pool")
        if s in ("median", "typical"):
            return sorted(sp, key=lambda t: t[1])[len(sp) // 2][0]
        if s.isdigit():
            return self.resolve(int(s))
        # task / model substring match
        hits = [i for i, p in enumerate(self.programs)
                if s in (p["task"] or "").lower() or s in (p["model"] or "").lower()
                or s in (p["type"] or "").lower()]
        if hits:
            return hits[0]
        raise ValueError("could not resolve %r" % selector)

    # ---- reads (the tool payloads) ----
    def read_program(self, selector) -> dict:
        i = self.resolve(selector)
        p = dict(self.programs[i])
        p["program_index"] = i
        p["note"] = ("every field is a real measurement from the frozen run pool; "
                     "reward_correctness_only is what an output-equivalence acceptance test pays; "
                     "reward_efficiency_gated = 1[correct]*min(S8/8,1); the serial_projection is a "
                     "static source transform (not re-executed).")
        return p

    def list_programs(self) -> list[dict]:
        return [{"program_index": i, "task": p["task"], "model": p["model"],
                 "condition": p["condition"], "self_speedup8": p["self_speedup8"],
                 "verdict": p["verdict"]} for i, p in enumerate(self.programs)]

    def scaling_distribution(self) -> dict:
        """The population view -- what the correctness-only reward cannot see across all accepted programs."""
        sp = [p["self_speedup8"] for p in self.programs if p["self_speedup8"] is not None]
        eff = [p["parallel_efficiency8"] for p in self.programs if p["parallel_efficiency8"] is not None]
        n = len(sp)
        # widest within-task spread (two programs, same task, certified equally correct)
        by_task = {}
        for p in self.programs:
            if p["self_speedup8"] is not None:
                by_task.setdefault(p["task"], []).append(p["self_speedup8"])
        spreads = [(t, min(v), max(v)) for t, v in by_task.items() if len(v) >= 2]
        worst = max(spreads, key=lambda z: z[2] - z[1]) if spreads else None
        return {
            "n_programs": n,
            "self_speedup8_min": round(min(sp), 2), "self_speedup8_max": round(max(sp), 2),
            "self_speedup8_median": round(_st.median(sp), 2),
            "below_serial_count": sum(x < 1.0 for x in sp),
            "below_serial_pct": round(100 * sum(x < 1.0 for x in sp) / n, 1),
            "below_2x_pct": round(100 * sum(x < 2.0 for x in sp) / n, 1),
            "efficiency_below_0.25_pct": round(100 * sum(e < 0.25 for e in eff) / len(eff), 1),
            "all_reward_correctness_only": 1.0,  # the point: every one of these scores identically
            "widest_within_task_spread": (
                {"task": worst[0], "min": round(worst[1], 2), "max": round(worst[2], 2)} if worst else None),
            "note": "all n programs are certified correct at every thread count and receive the "
                    "identical correctness-only reward of 1.0; the spread is invisible to it.",
        }

    def reward_comparison(self, selector) -> dict:
        """For one program: what each reward pays it, and what its serial projection would pay."""
        i = self.resolve(selector)
        p = self.programs[i]
        proj = p.get("serial_projection") or {}
        gated_serial_floor = round(1.0 / _NMAX, 3)
        return {
            "task": p["task"], "model": p["model"], "self_speedup8": p["self_speedup8"],
            "reward_correctness_only": p["reward_correctness_only"],
            "reward_efficiency_gated": p["reward_efficiency_gated"],
            "if_pragmas_stripped": {
                "predicted_still_correct": proj.get("predicted_still_correct"),
                "reward_correctness_only": (p["reward_correctness_only"]
                                            if proj.get("predicted_still_correct") else 0.0),
                "reward_efficiency_gated": (gated_serial_floor
                                            if proj.get("predicted_still_correct") else 0.0),
                "delta_correctness_only": 0.0 if proj.get("predicted_still_correct") else None,
            },
            "reading": ("under the correctness-only reward the stripped (serial) program scores "
                        "identically -- zero cost to delete the parallelism; under the gated reward "
                        "it drops to the 1/%d serial floor." % _NMAX),
        }


if __name__ == "__main__":
    s = Session()
    print("\nslowest-scaling accepted program:")
    p = s.read_program("slowest")
    print("  %s | %s | %s" % (p["task"], p["model"], p["verdict"]))
    print("  S(8)=%.2fx  eff=%.2f  R_corr=%.0f  R_gate=%.3f  race=%s  idiom=%s"
          % (p["self_speedup8"], p["parallel_efficiency8"], p["reward_correctness_only"],
             p["reward_efficiency_gated"], p["race_check"].get("clean"), p["sync_idiom"].get("primary")))
    print("  serial projection:", p["serial_projection"]["predicted_still_correct"],
          "-", p["serial_projection"]["reason"])
    print("\npopulation the correctness reward cannot see:")
    d = s.scaling_distribution()
    print("  ", {k: d[k] for k in ("n_programs", "self_speedup8_min", "self_speedup8_max",
                                    "below_serial_pct", "widest_within_task_spread")})
