"""Load TPC-H tables (generated natively by DuckDB) as pandas DataFrames.

Kept small by default so Spark's Arrow-less ``createDataFrame`` stays fast on a
laptop. Scale factor is the knob for the scaling/robustness experiments.
"""
from __future__ import annotations

from harness.config import get_duckdb

# Minimal column projections we actually use (keeps DataFrames light).
DEFAULT_COLUMNS = {
    "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"],
    "customer": ["c_custkey", "c_name", "c_nationkey"],
    "lineitem": ["l_orderkey", "l_linenumber", "l_extendedprice", "l_discount", "l_quantity"],
    "nation": ["n_nationkey", "n_name"],
}


def tpch_tables(sf: float = 0.03, tables=("orders",), columns: dict | None = None):
    columns = columns or DEFAULT_COLUMNS
    con = get_duckdb()
    con.sql(f"CALL dbgen(sf={sf})")
    out = {}
    for t in tables:
        cols = columns.get(t)
        sel = ", ".join(cols) if cols else "*"
        df = con.sql(f"SELECT {sel} FROM {t}").df()
        out[t] = df
    con.close()
    return out
