"""Real-benchmark per-entity daily time series derived from standard TPC-H.

We aggregate `lineitem` to per-supplier daily shipped quantity:

    entity_id = l_suppkey,  obs_date = l_shipdate,  value = SUM(l_quantity)

Gaps are ORGANIC: suppliers do not ship every day, so the series is naturally
irregular with staggered coverage -- nothing about the data is engineered to
trigger any particular bug. On this shape a *positional* offset (shift/LAG by k
rows) is not the same as a *calendar* offset (k days ago); at SF0.1 the two
disagree on ~98% of comparable rows.
"""
from __future__ import annotations

import pandas as pd

from harness.config import get_duckdb


def tpch_entity_daily(sf: float = 0.03) -> pd.DataFrame:
    con = get_duckdb()
    con.sql(f"CALL dbgen(sf={sf})")
    df = con.sql(
        """
        SELECT l_suppkey AS entity_id,
               l_shipdate AS obs_date,
               CAST(SUM(l_quantity) AS DOUBLE) AS value
        FROM lineitem
        GROUP BY l_suppkey, l_shipdate
        """
    ).df()
    con.close()
    df["entity_id"] = df["entity_id"].astype("int64")
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.normalize()
    return df.sort_values(["entity_id", "obs_date"]).reset_index(drop=True)


SCALE_TO_SF = {"small": 0.01, "base": 0.03, "large": 0.1, "xl": 0.3}
