"""Backfill a scheduled trigger.

Re-runs the slots a cron or fixed-rate trigger would have fired over a window,
naming each run exactly the way a real fire would so that slots which already ran
are recognised and skipped -- unless the backfill is forced, which salts the names
into a separate namespace and re-runs them.

The runs are created by a driver run inside the cluster, not from the client. See
``flyte backfill --help``.
"""

from ._driver import backfill_driver, launch_backfill
from ._execute import execute_plan, probe_existing
from ._naming import candidate_run_names, scheduled_run_name
from ._plan import DEFAULT_MAX_RUNS, BackfillPlan, BackfillSlot, build_plan, occurrences_for
from ._schedule import CronParseError, cron_occurrences, fixed_rate_occurrences

__all__ = [
    "DEFAULT_MAX_RUNS",
    "BackfillPlan",
    "BackfillSlot",
    "CronParseError",
    "backfill_driver",
    "build_plan",
    "candidate_run_names",
    "cron_occurrences",
    "execute_plan",
    "fixed_rate_occurrences",
    "launch_backfill",
    "occurrences_for",
    "probe_existing",
    "scheduled_run_name",
]
