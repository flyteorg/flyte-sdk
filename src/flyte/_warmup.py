"""Warm-up runs for reusable environments.

A warm-up run is flagged with a single run label. A backend that supports warm-up runs (Union)
handles a flagged action without dispatching it to the task's container: user code never runs,
no outputs are produced, and the action is not billed. For a task in a reusable environment
(`flyte.ReusePolicy`) the backend makes sure the environment exists, scaled to its minimum
replica count, and restarts its idle-TTL clock, so periodic warm-ups keep the pool alive. For
any other task a warm-up run succeeds without doing anything.

Backends that do not know the label run the task normally.
"""

from __future__ import annotations

from typing import Dict, Mapping

# Must match the label the backend checks (Union leaseworker and billing). Only the exact
# value "true" marks a warm-up run.
WARMUP_LABEL = "flyte.org/warmup"
WARMUP_LABEL_VALUE = "true"


def with_warmup_label(labels: Mapping[str, str] | None, warmup: bool) -> Dict[str, str] | None:
    """Return `labels` with the warm-up label added when `warmup` is set; unchanged otherwise."""
    if not warmup:
        return dict(labels) if labels is not None else None
    merged = dict(labels or {})
    merged[WARMUP_LABEL] = WARMUP_LABEL_VALUE
    return merged
