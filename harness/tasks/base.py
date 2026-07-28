"""Task interface: each task owns a prompt, an input generator, and a *trusted*
reference implementation (single-node pandas) used as the correctness oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class Task:
    id: str
    category: str
    title: str
    prompt: str                                  # natural-language spec handed to the agent
    make_inputs: Callable[[str], dict]           # scale -> {table_name: DataFrame}
    reference: Callable[[dict], pd.DataFrame]     # trusted single-node implementation
    key_cols: list[str]
    value_cols: list[str]
    input_tables: tuple[str, ...] = ()          # dict keys / view names given to the model
    output_contract: str = ""                     # exact output columns (given to agent)
    engines: tuple[str, ...] = ("duckdb", "spark")
    scales: tuple[str, ...] = ("base",)
    partitions: tuple[int, ...] = (1, 4, 16)      # Spark partition widths
    duckdb_threads: tuple[int, ...] = (1, 4)      # DuckDB thread counts
    repeats: int = 1                              # repeated runs (determinism check)
