"""Task: price-weighted average discount over all line items.

Category: distributed_aggregation (non-distributive aggregation)

Averages are not distributive: a price-weighted rate SUM(w*x)/SUM(w) is not the
same as the unweighted AVG(x). The natural WRONG answer computes AVG(l_discount);
the correct answer forms the ratio of sums. The result is partition-invariant
(sums are distributive), so the signature is "wrong-but-stable" -- a bug that
differential-across-partitions testing alone would miss, but the reference
oracle catches.
"""
from __future__ import annotations

from harness.data.gen_tpch import tpch_tables
from harness.tasks.base import Task

PROMPT = """You are given a distributed table `lineitem` with columns:
  l_extendedprice : DOUBLE
  l_discount      : DOUBLE

Task: compute the overall discount rate WEIGHTED by extended price, i.e.
  SUM(l_discount * l_extendedprice) / SUM(l_extendedprice)

Return a SINGLE row with exactly one column:
  weighted_avg_discount"""

OUTPUT_CONTRACT = "weighted_avg_discount (single row, single column)"


def _make_inputs(scale: str) -> dict:
    sf = {"base": 0.03, "small": 0.01, "large": 0.1}.get(scale, 0.03)
    t = tpch_tables(sf=sf, tables=("lineitem",))
    li = t["lineitem"][["l_extendedprice", "l_discount"]].copy()
    return {"lineitem": li}


def _reference(inputs: dict):
    import pandas as pd

    li = inputs["lineitem"]
    num = (li["l_discount"] * li["l_extendedprice"]).sum()
    den = li["l_extendedprice"].sum()
    return pd.DataFrame({"weighted_avg_discount": [float(num / den)]})


TASK = Task(
    id="weighted_avg_discount",
    category="distributed_aggregation",
    title="Price-weighted average discount (non-distributive aggregation)",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=[],
    value_cols=["weighted_avg_discount"],
    input_tables=("lineitem",),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=1,
)
