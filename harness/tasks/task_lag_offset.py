"""Task: per-entity calendar-offset lookup on a gapped time series.

Category: window_offset_per_key_lag

This is the public reproduction of a classic distribution-/data-shape-semantics
bug: the prompt asks for the value "28 calendar days ago", which reads naturally
as a fixed window offset (``LAG(value, 28)`` / ``shift(28)``). That positional
interpretation is correct ONLY on gap-free, one-row-per-day data. On realistic
gapped, staggered data it silently returns the wrong row and NULLs the first 28
observations of every key. The correct implementation is a *calendar* self-join.
"""
from __future__ import annotations

import pandas as pd

from harness.data.gen_timeseries import tpch_entity_daily, SCALE_TO_SF
from harness.tasks.base import Task

OFFSET_DAYS = 28

PROMPT = f"""You are given a distributed table `events` with columns:
  entity_id : BIGINT   -- identifier of an entity
  obs_date  : DATE     -- an observation date for that entity
  value     : DOUBLE   -- a measurement recorded on that date

Notes about the data:
  * Each entity starts being observed on a different date.
  * Not every calendar day is present for an entity (there are gaps / missing days).

Task: for each (entity_id, obs_date) in `events` such that the SAME entity also
has an observation exactly {OFFSET_DAYS} CALENDAR days earlier, output that
earlier observation's `value` as `value_28d_ago`. Omit any (entity_id, obs_date)
for which no observation exists exactly {OFFSET_DAYS} calendar days before it.

Output columns (one row per qualifying pair):
  entity_id, obs_date, value_28d_ago

Write efficient code suitable for a large distributed dataset."""

OUTPUT_CONTRACT = ("entity_id, obs_date, value_28d_ago "
                   "(one row per (entity_id, obs_date) that has an observation exactly 28 days earlier)")


def _make_inputs(scale: str) -> dict:
    df = tpch_entity_daily(sf=SCALE_TO_SF.get(scale, 0.03))
    return {"events": df}


def _reference(inputs: dict) -> pd.DataFrame:
    """Trusted calendar-offset implementation (single-node), well-posed:
    only pairs that HAVE an observation exactly OFFSET_DAYS earlier are emitted,
    so the output carries no structural nulls."""
    e = inputs["events"][["entity_id", "obs_date", "value"]].copy()
    src = e.rename(columns={"obs_date": "src_date", "value": "value_28d_ago"})
    src["obs_date"] = src["src_date"] + pd.Timedelta(days=OFFSET_DAYS)
    out = e.merge(src[["entity_id", "obs_date", "value_28d_ago"]],
                  on=["entity_id", "obs_date"], how="inner")
    return out[["entity_id", "obs_date", "value_28d_ago"]]


TASK = Task(
    id="lag_offset_28d",
    category="window_offset_per_key_lag",
    title="Per-entity 28-calendar-day offset on a gapped time series",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=["entity_id", "obs_date"],
    value_cols=["value_28d_ago"],
    input_tables=("events",),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=1,
)
