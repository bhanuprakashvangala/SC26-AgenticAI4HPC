"""Mechanism demonstration: the differential verifier catches exactly the failure
mode a single-thread check is blind to. We hand-write the *racy* completion a naive
(or weaker) model would produce for three ParEval tasks -- one per hazard class --
and verify each TWICE with the same tool used in the study:

  * execute-once  (validate at 1 thread)          -> should ACCEPT the racy code
  * differential  (validate across a thread sweep) -> should REJECT it

This is the positive control for the study's negative result: the two frontier
models simply did not emit code like this on standard tasks, so the verifier had
nothing to catch -- but when the failure mode is present, it is caught.

Run:  python -m harness.demo_injected_races
Emits results/injected_numbers.tex (macros) and prints a table.
"""
from __future__ import annotations
import json
from pathlib import Path

from harness.run_pareval_batch import Cell, batch_verify, _load_tasks, THREADS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "injected_races.jsonl"

# One injected race per hazard class. Each is CORRECT on a single thread and WRONG
# once the loop is split, for the reason named in `hazard`.
INJECTED = {
    "21_histogram_bin_0-100": dict(
        hazard="data race (unsynchronized shared-bin update)",
        code="""
void binsBy10Count(std::vector<double> const& x, std::array<size_t, 10> &bins) {
    bins.fill(0);
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        size_t bin = std::min(static_cast<size_t>(x[i] / 10), bins.size() - 1);
        bins[bin] += 1;            // RACE: concurrent unsynchronized increment
    }
}
"""),
    "29_reduce_sum_of_min_of_pairs": dict(
        hazard="data race (unsynchronized reduction accumulator)",
        code="""
double sumOfMinimumElements(std::vector<double> const& x, std::vector<double> const& y) {
    double sum = 0.0;
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::min(x[i], y[i]);   // RACE: no reduction clause
    }
    return sum;
}
"""),
    "38_search_find_the_first_even_number": dict(
        hazard="order-dependent selection (racy min-index)",
        code="""
size_t findFirstEven(std::vector<int> const& x) {
    size_t result = x.size();
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] % 2 == 0 && i < result) {
            result = i;            // RACE: check-then-write on shared result
        }
    }
    return result;
}
"""),
}


def main():
    tasks = _load_tasks()
    eo_cells, dv_cells, specs = [], [], {}
    for name, spec in INJECTED.items():
        t = tasks[name]
        eo = Cell(name, t, "injected", "execute_once", 0); eo.code = spec["code"]
        dv = Cell(name, t, "injected", "differential_verify", 0); dv.code = spec["code"]
        eo_cells.append(eo); dv_cells.append(dv); specs[name] = (spec, eo, dv)
    # Two VM calls total: all execute-once cells, then all differential cells.
    eo_res = batch_verify(eo_cells, [1], 1, "inj_eo")
    dv_res = batch_verify(dv_cells, THREADS, 3, "inj_dv")

    rows, macros = [], []
    print(f"\n{'task':34}{'hazard':44}{'exec-once':12}{'differential':12}")
    print("-" * 102)
    n_certified_but_racy = 0
    for name, (spec, eo, dv) in specs.items():
        eo_v = eo_res.get(eo.id, {}).get("verdict", "?")
        dvr = dv_res.get(dv.id, {})
        dv_v = dvr.get("verdict", "?")
        if eo_v == "ROBUST_CORRECT" and dv_v == "NOT_ROBUST":
            n_certified_but_racy += 1
        print(f"{name:34}{spec['hazard']:44}{eo_v:12}{dv_v:12}")
        rows.append(dict(task=name, hazard=spec["hazard"],
                         execute_once=eo_v, differential=dv_v, diag=dvr.get("diag", "")))
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    macros.append(f"\\newcommand{{\\Ninjected}}{{{len(INJECTED)}}}")
    macros.append(f"\\newcommand{{\\NinjectedCaught}}{{{n_certified_but_racy}}}")
    (ROOT / "results" / "injected_numbers.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8")
    print(f"\n{n_certified_but_racy}/{len(INJECTED)} injected races were CERTIFIED by "
          f"execute-once but CAUGHT by the differential verifier.")
    print("wrote results/injected_races.jsonl and results/injected_numbers.tex")


if __name__ == "__main__":
    main()
