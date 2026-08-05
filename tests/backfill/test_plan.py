"""Planning decides what gets created, skipped, or re-run."""

from datetime import datetime, timezone

import pytest

from flyte.backfill._naming import scheduled_run_name
from flyte.backfill._plan import DEFAULT_MAX_RUNS, build_plan

ORG, PROJECT, DOMAIN = "acme", "ml-platform", "production"
TASK, TRIGGER = "evals.weekly.run", "nightly_eval"
START = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 5, 10, 23, 59, tzinfo=timezone.utc)


class _FakeCron:
    def __init__(self, expression="0 2 * * *", timezone_name="UTC"):
        self.expression = expression
        self.timezone = timezone_name


class _FakeSchedule:
    def __init__(self, expression="0 2 * * *", kickoff=""):
        self._cron = _FakeCron(expression)
        self.cron = self._cron
        self.cron_expression = ""
        self.kickoff_time_input_arg = kickoff

    def HasField(self, field):
        return field == "cron"


class _FakeName:
    def __init__(self, name):
        self.name = name


class _FakeId:
    def __init__(self, name):
        self.name = _FakeName(name)


class _FakePb2:
    def __init__(self, expression="0 2 * * *", kickoff=""):
        self.id = _FakeId(TRIGGER)
        self.automation_spec = type("A", (), {"schedule": _FakeSchedule(expression, kickoff)})()


class FakeTriggerDetails:
    """Stands in for a remote TriggerDetails, exposing only what planning reads."""

    def __init__(self, expression="0 2 * * *", kickoff=""):
        self.pb2 = _FakePb2(expression, kickoff)


def _plan(**overrides):
    kwargs = {
        "details": FakeTriggerDetails(),
        "task_name": TASK,
        "org": ORG,
        "project": PROJECT,
        "domain": DOMAIN,
        "start": START,
        "end": END,
        "max_runs": DEFAULT_MAX_RUNS,
    }
    kwargs.update(overrides)
    return build_plan(**kwargs)


def test_one_slot_per_scheduled_fire():
    plan = _plan()
    assert len(plan.slots) == 10  # daily at 02:00, May 1-10
    assert all(s.scheduled_at.hour == 2 for s in plan.slots)


def test_slot_names_match_a_real_scheduled_fire():
    plan = _plan()
    first = plan.slots[0]
    assert first.run_name == scheduled_run_name(ORG, PROJECT, DOMAIN, TASK, TRIGGER, first.scheduled_at)


def test_existing_runs_are_skipped():
    plan = _plan()
    already = plan.slots[3].run_name
    replanned = _plan(existing=[already])
    assert len(replanned.skipped) == 1
    assert len(replanned.to_create) == 9
    assert replanned.skipped[0].run_name == already


def test_existing_is_matched_against_the_rewritten_prefix_too():
    """A scheduled run routed to actions is stored under a 'u' prefix."""
    plan = _plan()
    stored_as = "u" + plan.slots[0].run_name[1:]
    replanned = _plan(existing=[stored_as])
    assert len(replanned.skipped) == 1


def test_force_recreates_existing_slots_under_salted_names():
    plan = _plan()
    already = plan.slots[0].run_name
    forced = _plan(existing=[already], force=True)
    assert len(forced.to_create) == 10  # nothing skipped
    assert len(forced.overridden) == 1
    # The forced name must differ, or it would be de-duplicated against the original.
    assert forced.slots[0].run_name != already


def test_suffix_distinguishes_repeated_forced_backfills():
    first = _plan(force=True, suffix="rerun-1")
    second = _plan(force=True, suffix="rerun-2")
    assert first.slots[0].run_name != second.slots[0].run_name


def test_plan_is_capped_and_reports_what_it_dropped():
    plan = _plan(end=datetime(2026, 8, 1, tzinfo=timezone.utc), max_runs=10)
    assert len(plan.slots) == 10
    assert plan.truncated > 0


def test_start_must_precede_end():
    with pytest.raises(ValueError):
        _plan(start=END, end=START)


def test_trigger_without_a_schedule_is_rejected():
    details = FakeTriggerDetails()
    details.pb2.automation_spec.schedule.cron_expression = ""
    details.pb2.automation_spec.schedule.HasField = lambda field: False
    with pytest.raises(ValueError, match="no schedule"):
        _plan(details=details)
