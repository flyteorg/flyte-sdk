"""Tests for the rich-repr rendering path used by `flyte get trigger`."""

from datetime import datetime

import pytest
from flyteidl2.task import common_pb2
from flyteidl2.trigger import trigger_definition_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from flyte.cli._common import format
from flyte.remote._trigger import Trigger


def _timestamp(dt: datetime) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def _trigger(automation: common_pb2.TriggerAutomationSpec | None, name: str = "t") -> Trigger:
    pb2 = trigger_definition_pb2.Trigger(active=True)
    pb2.id.name.name = name
    pb2.id.name.task_name = "my_task"
    if automation is not None:
        pb2.automation_spec.CopyFrom(automation)
    return Trigger(pb2=pb2)


def _schedule(**kwargs) -> common_pb2.TriggerAutomationSpec:
    return common_pb2.TriggerAutomationSpec(
        type=common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE,
        schedule=common_pb2.Schedule(**kwargs),
    )


@pytest.mark.parametrize(
    "automation, expected",
    [
        pytest.param(None, "TYPE_UNSPECIFIED", id="unset"),
        pytest.param(
            common_pb2.TriggerAutomationSpec(type=common_pb2.TriggerAutomationSpecType.TYPE_NONE),
            "none",
            id="none",
        ),
        pytest.param(
            _schedule(cron=common_pb2.Cron(expression="0 * * * *", timezone="US/Pacific")),
            "cron: 0 * * * * (US/Pacific)",
            id="cron",
        ),
        pytest.param(
            _schedule(cron=common_pb2.Cron(expression="0 * * * *")),
            "cron: 0 * * * * (UTC)",
            id="cron-default-timezone",
        ),
        pytest.param(
            _schedule(cron_expression="*/5 * * * *"),
            "cron: */5 * * * *",
            id="cron-expression",
        ),
        pytest.param(
            _schedule(rate=common_pb2.FixedRate(value=60, unit=common_pb2.FIXED_RATE_UNIT_MINUTE)),
            "every 60 minutes starting at now",
            id="fixed-rate",
        ),
        pytest.param(
            _schedule(
                rate=common_pb2.FixedRate(
                    value=1,
                    unit=common_pb2.FIXED_RATE_UNIT_DAY,
                    start_time=_timestamp(datetime(2026, 7, 29, 17, 0, 0)),
                )
            ),
            "every 1 day starting at 2026-07-29 17:00:00",
            id="fixed-rate-with-start-time",
        ),
        pytest.param(
            common_pb2.TriggerAutomationSpec(type=common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE),
            "schedule: unset",
            id="schedule-without-expression",
        ),
    ],
)
def test_rich_repr_automation(automation, expected):
    assert list(_trigger(automation).__rich_repr__()) == [
        ("task_name", "my_task"),
        ("name", "t"),
        ("automation", expected),
        ("auto_activate", True),
    ]


def test_format_table_columns_are_stable_across_automation_kinds():
    """Triggers with different automations must share one column set, or rows misalign."""
    triggers = [
        _trigger(common_pb2.TriggerAutomationSpec(type=common_pb2.TriggerAutomationSpecType.TYPE_NONE), name="a"),
        _trigger(_schedule(cron=common_pb2.Cron(expression="0 * * * *")), name="b"),
        _trigger(_schedule(rate=common_pb2.FixedRate(value=5, unit=common_pb2.FIXED_RATE_UNIT_HOUR)), name="c"),
    ]

    table = format("Triggers", triggers)

    assert [c.header for c in table.columns] == ["Task_name", "Name", "Automation", "Auto_activate"]
    assert list(table.columns[2].cells) == [
        "none",
        "cron: 0 * * * * (UTC)",
        "every 5 hours starting at now",
    ]
