"""Synthetic *public* per-entity event stream with irregular gaps.

This is the open, reproducible stand-in for a very common real-world shape:
per-key time series where each key starts on a different date and has missing
days (gaps). On data of this shape, a *positional* window offset
(``LAG(value, 28)`` / ``shift(28)``) is NOT the same as a *calendar* offset
("the value 28 days ago"): the positional version silently returns the wrong
row whenever gaps exist, and NULLs the first 28 observations of every key.

Nothing here is domain-specific; it is generated from a fixed seed so the whole
study is reproducible on any machine.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd


def make_events(
    n_entities: int = 400,
    horizon_days: int = 180,
    seed: int = 20260801,
    min_start_offset: int = 0,
    max_start_offset: int = 60,
    keep_prob_range: tuple[float, float] = (0.55, 0.9),
) -> pd.DataFrame:
    """Return a DataFrame [entity_id, obs_date, value] with staggered starts and gaps.

    * Each entity starts on ``base_start + U(min_start_offset, max_start_offset)``
      days, so keys are not aligned (mirrors different onboarding dates).
    * Each calendar day in an entity's active window is *kept* with probability
      drawn per-entity from ``keep_prob_range`` -> irregular gaps.
    * ``value`` is a smooth per-entity random walk so that "28 days ago" is a
      meaningful, checkable quantity.
    """
    rng = np.random.default_rng(seed)
    base_start = date(2025, 1, 1)
    end = base_start + timedelta(days=horizon_days)

    rows = []
    for e in range(n_entities):
        start = base_start + timedelta(days=int(rng.integers(min_start_offset, max_start_offset + 1)))
        keep_p = float(rng.uniform(*keep_prob_range))
        level = float(rng.uniform(10, 100))
        d = start
        while d <= end:
            if rng.uniform() <= keep_p:
                level += float(rng.normal(0, 3))
                rows.append((f"E{e:04d}", d, round(max(level, 0.0), 3)))
            d += timedelta(days=1)

    df = pd.DataFrame(rows, columns=["entity_id", "obs_date", "value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df.sort_values(["entity_id", "obs_date"]).reset_index(drop=True)


def divergence_report(df: pd.DataFrame, offset_days: int = 28) -> dict:
    """Quantify how often positional-shift disagrees with calendar-offset.

    This is a *ground-truth* diagnostic (not part of any candidate solution): it
    shows the bug is actually triggered by the data shape, and by how much.
    """
    d = df.sort_values(["entity_id", "obs_date"]).copy()
    # Naive positional offset (the common bug):
    d["pos_shift"] = d.groupby("entity_id")["value"].shift(offset_days)
    # Correct calendar offset via self-join on date - offset:
    left = d[["entity_id", "obs_date", "value"]].copy()
    right = d[["entity_id", "obs_date", "value"]].rename(
        columns={"obs_date": "src_date", "value": "cal_offset"}
    )
    right["obs_date"] = right["src_date"] + pd.Timedelta(days=offset_days)
    merged = left.merge(right[["entity_id", "obs_date", "cal_offset"]], on=["entity_id", "obs_date"], how="left")
    d = d.merge(merged[["entity_id", "obs_date", "cal_offset"]], on=["entity_id", "obs_date"], how="left")

    both = d.dropna(subset=["pos_shift", "cal_offset"])
    disagree = (~np.isclose(both["pos_shift"], both["cal_offset"])).sum()
    return {
        "rows": int(len(d)),
        "entities": int(d["entity_id"].nunique()),
        "rows_both_defined": int(len(both)),
        "rows_pos_null_cal_defined": int(d["pos_shift"].isna().sum() - d["cal_offset"].isna().sum()
                                         if d["pos_shift"].isna().sum() >= d["cal_offset"].isna().sum() else 0),
        "disagreements": int(disagree),
        "disagreement_rate": round(float(disagree) / max(len(both), 1), 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=400)
    ap.add_argument("--horizon-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    df = make_events(args.n_entities, args.horizon_days, args.seed)
    rep = divergence_report(df, 28)
    print("events:", rep)
    if args.out:
        df.to_csv(args.out, index=False)
        print("wrote", args.out, len(df), "rows")
