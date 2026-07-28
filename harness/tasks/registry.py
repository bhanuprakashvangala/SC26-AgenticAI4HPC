"""Registry of all evaluation tasks.

Note: the lag_offset task was removed from the active study (its reference was
degenerate -- 78% null -- and its round-0 candidate outputs were lost to an
infrastructure error). The task module is retained for future re-runs but is not
part of the reported study.
"""
from __future__ import annotations

from harness.tasks.task_first_order import TASK as _first
from harness.tasks.task_customer_order_counts import TASK as _counts
from harness.tasks.task_stable_id import TASK as _sid
from harness.tasks.task_weighted_avg import TASK as _wavg

ALL_TASKS = {t.id: t for t in (_first, _counts, _sid, _wavg)}


def get_task(task_id: str):
    return ALL_TASKS[task_id]
