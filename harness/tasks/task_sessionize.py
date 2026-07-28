"""Task: count activity sessions per user (gap-and-islands / sessionization).

Category: order_dependent_state

A new session starts at a user's first event and whenever the gap from the
previous event (in chronological order, per user) exceeds 30 minutes. The correct
implementation imposes a TOTAL order per user -- Window.partitionBy(user).orderBy(ts),
lag(ts), then a running count of gap>30 boundaries. The input is deliberately
shuffled, so any implementation that does not order events within each user (or
that computes the gap globally) produces a partition/order-dependent, wrong count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness.tasks.base import Task

PROMPT = """You are given a distributed table `events` with columns:
  user_id : BIGINT
  ts      : TIMESTAMP   (the time of one activity event)

Each user's events form SESSIONS. Scanning a user's events in chronological
order, a new session begins at their first event and again whenever the gap from
that user's previous event is STRICTLY MORE THAN 30 minutes.

Task: for each user, count how many distinct sessions they have.

Output exactly one row per user, with columns:
  user_id, n_sessions"""

OUTPUT_CONTRACT = "user_id, n_sessions (one row per user)"


def _make_inputs(scale: str) -> dict:
    rng = np.random.default_rng(7)
    nusers = {"base": 400, "small": 150, "large": 1200}.get(scale, 400)
    base = pd.Timestamp("2025-01-01")
    rows = []
    for u in range(1, nusers + 1):
        n = int(rng.integers(3, 20))
        t = base + pd.Timedelta(minutes=int(rng.integers(0, 10000)))
        for _ in range(n):
            rows.append((u, t))
            # within-session gaps (<30) vs between-session gaps (>30); never ==30
            gap = int(rng.integers(1, 29)) if rng.random() < 0.5 else int(rng.integers(31, 240))
            t = t + pd.Timedelta(minutes=gap)
    df = pd.DataFrame(rows, columns=["user_id", "ts"])
    return {"events": df.sample(frac=1.0, random_state=1).reset_index(drop=True)}


def _reference(inputs: dict) -> pd.DataFrame:
    df = inputs["events"].sort_values(["user_id", "ts"], kind="mergesort").copy()
    prev = df.groupby("user_id")["ts"].shift(1)
    gap_min = (df["ts"] - prev).dt.total_seconds() / 60.0
    df["new_session"] = prev.isna() | (gap_min > 30.0)
    out = df.groupby("user_id", as_index=False)["new_session"].sum()
    out.columns = ["user_id", "n_sessions"]
    out["n_sessions"] = out["n_sessions"].astype("int64")
    return out


TASK = Task(
    id="sessionize_events",
    category="order_dependent_state",
    title="Sessionization by 30-minute inactivity gap",
    prompt=PROMPT,
    make_inputs=_make_inputs,
    reference=_reference,
    key_cols=["user_id"],
    value_cols=["n_sessions"],
    input_tables=("events",),
    output_contract=OUTPUT_CONTRACT,
    engines=("duckdb", "spark"),
    scales=("base",),
    partitions=(1, 4, 16),
    duckdb_threads=(1, 4),
    repeats=2,
)
