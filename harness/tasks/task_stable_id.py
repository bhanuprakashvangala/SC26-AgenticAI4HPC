"""Task: assign a reproducible dense surrogate key over a total order.

Category: nondeterministic_id (partition-dependent identifiers)

The natural WRONG answer uses a partition-dependent id source
(`monotonically_increasing_id()`, `zipWithIndex` on an unordered RDD, or
`row_number()` over an unspecified order). Those ids change with partition width
and do not form the requested contiguous 0..N-1 sequence. The correct answer
imposes the requested TOTAL order and densely numbers from 0.
"""
from __future__ import annotations

import pandas as pd

from harness.data.gen_events import make_events
from harness.data.gen_timeseries import tpch_entity_daily, SCALE_TO_SF
from harness.tasks.base import Task

PROMPT = """You are given a distributed table `events` with columns:
  entity_id : BIGINT
  obs_date  : DATE
  value     : DOUBLE
Each (entity_id, obs_date) pair is unique.

Task: assign every row a surrogate key `sk`, a unique integer in the range
0..N-1 (N = number of rows), assigned in ascending order of (entity_id, obs_date):
the globally smallest (entity_id, obs_date) gets sk = 0, the next gets 1, and so
on with no gaps. The assignment must be reproducible.

Output exactly one row per input row, with columns:
  entity_id, obs_date, sk"""

OUTPUT_CONTRACT = "entity_id, obs_date, sk (dense 0..N-1 in ascending (entity_id, obs_date) order)"


def _make_inputs(scale: str) -> dict:
    df = tpch_entity_daily(sf=SCALE_TO_SF.get(scale, 0.03))
    return {"events": df}


def _reference(inputs: dict) -> pd.DataFrame:
    e = inputs["events"][["entity_id", "obs_date", "value"]].copy()
    e = e.sort_values(["entity_id", "obs_date"], kind="mergesort").reset_index(drop=True)
    e["sk"] = range(len(e))
    return e[["entity_id", "obs_date", "sk"]]


TASK = Task(
    id="stable_surrogate_key",
    category="nondeterministic_id",
    title="Reproducible dense surrogate key over a total order",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=["entity_id", "obs_date"],
    value_cols=["sk"],
    input_tables=("events",),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=2,
)
