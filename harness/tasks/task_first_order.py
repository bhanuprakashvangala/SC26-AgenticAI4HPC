"""Task: earliest order per customer, with a deterministic tie-break.

Category: nondeterministic_pick (partition/shuffle-order dependence)

The prompt asks for a single representative row per key. The natural but WRONG
implementations select an arbitrary representative (`dropDuplicates`, unordered
`first()`/`any_value`, `row_number` without a total order). Those depend on how
the data is partitioned, so results change with Spark partition width and rarely
match a deterministic reference. The correct implementation imposes a TOTAL order
(earliest o_orderdate, then smallest o_orderkey).
"""
from __future__ import annotations

import pandas as pd

from harness.data.gen_tpch import tpch_tables
from harness.tasks.base import Task

PROMPT = """You are given a distributed table `orders` with columns:
  o_orderkey  : BIGINT
  o_custkey   : BIGINT
  o_orderdate : DATE
  o_totalprice: DOUBLE

Task: for each customer (o_custkey), find their FIRST order, defined as the order
with the earliest o_orderdate. If a customer has several orders on that earliest
date, pick the one with the smallest o_orderkey.

Output exactly one row per customer, with columns:
  o_custkey, first_orderkey"""

OUTPUT_CONTRACT = "o_custkey, first_orderkey (one row per customer)"


def _make_inputs(scale: str) -> dict:
    sf = {"base": 0.03, "small": 0.01, "large": 0.1}.get(scale, 0.03)
    t = tpch_tables(sf=sf, tables=("orders",))
    df = t["orders"][["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]].copy()
    df["o_orderdate"] = pd.to_datetime(df["o_orderdate"]).dt.normalize()
    return {"orders": df}


def _reference(inputs: dict) -> pd.DataFrame:
    o = inputs["orders"].sort_values(["o_custkey", "o_orderdate", "o_orderkey"], kind="mergesort")
    first = o.groupby("o_custkey", as_index=False).first()
    return first[["o_custkey", "o_orderkey"]].rename(columns={"o_orderkey": "first_orderkey"})


TASK = Task(
    id="first_order_per_customer",
    category="nondeterministic_pick",
    title="Earliest order per customer with deterministic tie-break",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=["o_custkey"],
    value_cols=["first_orderkey"],
    input_tables=("orders",),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=2,
)
