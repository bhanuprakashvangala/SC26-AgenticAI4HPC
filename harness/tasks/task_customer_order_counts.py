"""Task: customer order counts including customers with zero orders.

Category: silent_row_loss (join semantics)

Two classic distribution-scale bugs hide here:
  * an INNER join (or grouping only the `orders` side) silently DROPS customers
    with no orders -> missing rows that no one notices at scale; and
  * a LEFT join followed by COUNT(*) reports 1 instead of 0 for zero-order
    customers (COUNT(*) counts the null-filled row).
The correct answer keeps every customer and counts non-null order keys.
"""
from __future__ import annotations

from harness.data.gen_tpch import tpch_tables
from harness.tasks.base import Task

PROMPT = """You are given two distributed tables.

`customer` with columns:
  c_custkey : BIGINT
  c_name    : STRING

`orders` with columns:
  o_orderkey : BIGINT
  o_custkey  : BIGINT

Task: report the number of orders for EVERY customer. Customers that have no
orders at all must still appear, with order_count = 0.

Output exactly one row per customer, with columns:
  c_custkey, order_count"""

OUTPUT_CONTRACT = "c_custkey, order_count (one row per customer, zero-order customers included)"


def _make_inputs(scale: str) -> dict:
    sf = {"base": 0.03, "small": 0.01, "large": 0.1}.get(scale, 0.03)
    t = tpch_tables(sf=sf, tables=("orders", "customer"))
    cust = t["customer"][["c_custkey", "c_name"]].copy()
    orders = t["orders"][["o_orderkey", "o_custkey"]].copy()
    return {"customer": cust, "orders": orders}


def _reference(inputs: dict):
    cust = inputs["customer"][["c_custkey"]].copy()
    orders = inputs["orders"]
    cnt = orders.groupby("o_custkey", as_index=False)["o_orderkey"].count()
    cnt = cnt.rename(columns={"o_custkey": "c_custkey", "o_orderkey": "order_count"})
    out = cust.merge(cnt, on="c_custkey", how="left")
    out["order_count"] = out["order_count"].fillna(0).astype("int64")
    return out[["c_custkey", "order_count"]]


TASK = Task(
    id="customer_order_counts",
    category="silent_row_loss",
    title="Order counts for every customer (including zero-order customers)",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=["c_custkey"],
    value_cols=["order_count"],
    input_tables=("customer", "orders"),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=1,
)
