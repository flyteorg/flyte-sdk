"""Work out which runs a backfill would create, before creating any of them.

The plan is built once and then either printed (``--dry-run``), shown for
confirmation, or handed to the driver to execute. Keeping it a plain data
structure means the CLI preview and the driver agree on exactly what will happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from ._naming import candidate_run_names, scheduled_run_name
from ._schedule import cron_occurrences, fixed_rate_occurrences

if TYPE_CHECKING:
    from flyte.remote import TriggerDetails

__all__ = ["DEFAULT_MAX_RUNS", "FORCE_SALT_PREFIX", "BackfillPlan", "BackfillSlot", "build_plan"]

# The console caps a backfill at this many runs; the CLI matches it so both
# surfaces behave the same. Wider ranges are a deliberate, explicit choice.
DEFAULT_MAX_RUNS = 100

# Prepended to salted (forced) names so they occupy a namespace that can never
# collide with a real scheduled fire.
FORCE_SALT_PREFIX = "bf1"


@dataclass(frozen=True)
class BackfillSlot:
    """One scheduled time and the run it maps to."""

    scheduled_at: datetime
    run_name: str
    #: True when a run for this slot already exists, i.e. the slot already fired.
    already_ran: bool = False

    @property
    def candidates(self) -> tuple[str, ...]:
        return candidate_run_names(self.run_name)


@dataclass
class BackfillPlan:
    """Everything a backfill will do, decided up front."""

    trigger_name: str
    task_name: str
    project: str
    domain: str
    org: str
    schedule: str
    start: datetime
    end: datetime
    force: bool
    salt: str | None
    queue: str | None
    max_runs: int
    slots: list[BackfillSlot] = field(default_factory=list)
    #: Slots dropped because the plan hit ``max_runs``.
    truncated: int = 0

    @property
    def to_create(self) -> list[BackfillSlot]:
        return [s for s in self.slots if self.force or not s.already_ran]

    @property
    def skipped(self) -> list[BackfillSlot]:
        return [s for s in self.slots if not self.force and s.already_ran]

    @property
    def overridden(self) -> list[BackfillSlot]:
        """Slots that already ran and will be re-run because ``force`` is set."""
        return [s for s in self.slots if self.force and s.already_ran]


def _schedule_expression(details: "TriggerDetails") -> tuple[str, str | None, int | None, datetime | None]:
    """Pull the schedule out of a trigger.

    Returns ``(human_expression, cron_expression, interval_minutes, rate_anchor)``.
    Exactly one of ``cron_expression`` / ``interval_minutes`` is set.
    """
    automation = details.pb2.automation_spec
    schedule = automation.schedule
    if schedule.HasField("cron"):
        cron = schedule.cron
        tz = cron.timezone or "UTC"
        return (f"{cron.expression} ({tz})", cron.expression, None, None)
    if schedule.cron_expression:  # deprecated form, still accepted server-side
        return (schedule.cron_expression, schedule.cron_expression, None, None)
    if schedule.HasField("rate"):
        rate = schedule.rate
        # FixedRate.unit is an enum of MINUTE/HOUR/DAY; normalise to minutes.
        unit_minutes = {0: 1, 1: 60, 2: 1440}.get(int(rate.unit), 1)
        minutes = int(rate.value) * unit_minutes
        anchor = rate.start_time.ToDatetime().replace(tzinfo=timezone.utc) if rate.HasField("start_time") else None
        return (f"every {minutes}m", None, minutes, anchor)
    raise ValueError(f"trigger {details.pb2.id.name!r} has no schedule -- only scheduled triggers can be backfilled")


def schedule_timezone(details: "TriggerDetails") -> str:
    schedule = details.pb2.automation_spec.schedule
    if schedule.HasField("cron") and schedule.cron.timezone:
        return schedule.cron.timezone
    return "UTC"


def occurrences_for(
    details: "TriggerDetails",
    start: datetime,
    end: datetime,
    limit: int | None = None,
) -> list[datetime]:
    """Expand a trigger's schedule across ``[start, end]``."""
    _, cron_expr, interval, anchor = _schedule_expression(details)
    if cron_expr is not None:
        return cron_occurrences(cron_expr, start, end, limit)
    assert interval is not None
    return fixed_rate_occurrences(interval, start, end, anchor=anchor, limit=limit)


def build_plan(
    *,
    details: "TriggerDetails",
    task_name: str,
    org: str,
    project: str,
    domain: str,
    start: datetime,
    end: datetime,
    force: bool = False,
    suffix: str | None = None,
    queue: str | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    existing: Sequence[str] | None = None,
) -> BackfillPlan:
    """Build the plan for backfilling ``details`` across ``[start, end]``.

    ``existing`` is the set of run names already known to exist; slots naming one
    of those are marked as already run (and skipped unless ``force``).
    """
    if start >= end:
        raise ValueError("start must be before end")

    trigger_name = details.pb2.id.name.name or details.pb2.id.name
    human_schedule, _, _, _ = _schedule_expression(details)
    salt = f"{FORCE_SALT_PREFIX}:{suffix}" if force and suffix else (FORCE_SALT_PREFIX if force else None)

    # Ask for one more than the cap so we can report how much was left out.
    raw = occurrences_for(details, start, end, limit=max_runs + 1 if max_runs else None)
    truncated = max(0, len(raw) - max_runs) if max_runs else 0
    known = set(existing or ())

    slots: list[BackfillSlot] = []
    for at in raw[:max_runs] if max_runs else raw:
        name = scheduled_run_name(org, project, domain, task_name, trigger_name, at, salt=salt)
        # Existence is always checked against the *unsalted* name -- that is the
        # one a real scheduled fire would have produced.
        unsalted = scheduled_run_name(org, project, domain, task_name, trigger_name, at)
        already = any(c in known for c in candidate_run_names(unsalted))
        slots.append(BackfillSlot(scheduled_at=at, run_name=name, already_ran=already))

    return BackfillPlan(
        trigger_name=trigger_name,
        task_name=task_name,
        project=project,
        domain=domain,
        org=org,
        schedule=human_schedule,
        start=start,
        end=end,
        force=force,
        salt=salt,
        queue=queue,
        max_runs=max_runs,
        slots=slots,
        truncated=truncated,
    )
