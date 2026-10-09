"""Colocated half of "lineage graphs and factories are functionally equivalent" (flyteplugins-union
docs/lineage-factory-parity.md): artifacts.partition, artifacts.required, the implicit partition rule and sinks
(refresh policies are in test_refresh.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts._lineage import BINDINGS_LABEL, declared_output_metadata, extract_task_lineage
from flyte.errors import LineageDeclarationError
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(name="parity")
report = artifacts.Artifact("pr_report", type=File, partitions={"date": artifacts.Daily})
events = artifacts.Artifact("pr_events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
flat = artifacts.Artifact("pr_flat", type=File)


def _params(t):
    return {
        k: {kk: vv for kk, vv in v.items() if kk not in ("src_file", "src_line")}
        for k, v in _b(t)["parameters"].items()
    }


def _b(t):
    return json.loads(extract_task_lineage(t).labels[BINDINGS_LABEL])


# ------------------------------------------------------------------ artifacts.partition


def test_partition_records_like_get_partition_value():
    async def f(week: list[DataFrame], day: datetime) -> File:
        raise NotImplementedError

    a = env.task(
        produces_artifacts=(report,),
        consumes_artifacts={"week": events.all("region"), "day": artifacts.partition("date")},
    )(f)
    b = env.task(
        produces_artifacts=(report,),
        consumes_artifacts={"week": events.all("region"), "day": report.get_partition_value("date")},
    )(f)
    assert _params(a)["day"] == {"kind": "partition", "node": "pr_report", "dim": "date", "type": "datetime"}
    assert _params(a) == _params(b)
    assert extract_task_lineage(a).pullable


def test_partition_needs_a_produced_dimension():
    async def f(region: str) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(report,), consumes_artifacts={"region": artifacts.partition("region")})(f)
    with pytest.raises(LineageDeclarationError, match=r"artifacts.partition\('region'\), but no artifact it produces"):
        extract_task_lineage(t)


def test_partition_type_must_carry_the_dimension():
    async def f(day: int) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(report,), consumes_artifacts={"day": artifacts.partition("date")})(f)
    with pytest.raises(LineageDeclarationError, match="must be typed datetime or date; it is int"):
        extract_task_lineage(t)


def test_partition_of_a_sink_reads_an_identity_input():
    async def f(ev: DataFrame, day: datetime) -> None:
        raise NotImplementedError

    t = env.task(consumes_artifacts={"ev": events.select(region="us"), "day": artifacts.partition("date")})(f)
    assert _params(t)["day"] == {"kind": "partition", "node": "pr_events", "dim": "date", "type": "datetime"}

    async def g(ev: DataFrame, r: str) -> None:
        raise NotImplementedError

    pinned = env.task(consumes_artifacts={"ev": events.select(region="us"), "r": artifacts.partition("region")})(g)
    with pytest.raises(LineageDeclarationError, match="it produces nothing"):
        extract_task_lineage(pinned)


def test_partition_at_run_time():
    async def f(day: datetime) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(report,), consumes_artifacts={"day": artifacts.partition("date")})(f)
    (decl,) = declared_output_metadata(t, {"day": datetime(2026, 9, 8, 13)}).values()
    assert decl.skip_reason is None and decl.metadata.partitions["date"].value == datetime(
        2026, 9, 8, tzinfo=timezone.utc
    )


# ------------------------------------------------------------------ artifacts.required


def test_required_is_recorded_and_plannable():
    async def f(date: datetime, seed: int) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(report,), consumes_artifacts={"seed": artifacts.required()})(f)
    lin = extract_task_lineage(t)
    assert _params(t)["seed"] == {"kind": "required", "type": "int"}
    assert lin.pullable and lin.warnings() == [] and lin.unpullable_params == []

    async def g(date: datetime, seed: int) -> File:
        raise NotImplementedError

    unmarked = env.task(produces_artifacts=(report,))(g)
    lin = extract_task_lineage(unmarked)
    assert not lin.pullable and lin.unpullable_params == ["seed"]
    assert "artifacts.required()" in lin.warnings()[0]


def test_required_marker_forms():
    assert artifacts.required().to_dict() == {"required": True}
    assert artifacts.partition("date").to_dict() == {"partition": "date"}
    with pytest.raises(ValueError):
        artifacts.partition("")


# ------------------------------------------------------------------ the implicit rule


def test_implicit_partition_param():
    async def f(date: datetime, region: str, n: int = 3) -> File:
        raise NotImplementedError

    two = artifacts.Artifact("pr_two", type=File, partitions={"date": artifacts.Daily, "region": str})
    t = env.task(produces_artifacts=(two,))(f)
    p = _params(t)
    assert p["date"] == {"kind": "partition", "node": "pr_two", "dim": "date", "type": "datetime", "implicit": True}
    assert p["region"]["implicit"] is True and p["n"]["kind"] == "default"
    assert extract_task_lineage(t).pullable
    (decl,) = declared_output_metadata(t, {"date": datetime(2026, 9, 8), "region": "us", "n": 3}).values()
    assert decl.metadata.partitions["region"] == "us"


def test_implicit_rule_leaves_bound_defaulted_and_mistyped_params_alone():
    async def f(date: datetime = datetime(2026, 1, 1)) -> File:
        raise NotImplementedError

    defaulted = env.task(produces_artifacts=(report,))(f)
    assert _params(defaulted)["date"]["kind"] == "default"
    assert declared_output_metadata(defaulted, {"date": datetime(2026, 9, 8)})["o0"].skip_reason

    async def g(date: str) -> File:
        raise NotImplementedError

    mistyped = env.task(produces_artifacts=(report,))(g)
    assert _params(mistyped)["date"]["kind"] == "unbound"

    async def h(date: datetime, day: datetime) -> File:
        raise NotImplementedError

    bound = env.task(produces_artifacts=(report,), consumes_artifacts={"date": artifacts.required()})(h)
    assert _params(bound)["date"]["kind"] == "required"


def test_implicit_rule_for_a_sink_uses_identity_inputs():
    async def f(ev: DataFrame, date: datetime, region: str) -> None:
        raise NotImplementedError

    t = env.task(consumes_artifacts={"ev": events})(f)
    p = _params(t)
    assert p["date"]["implicit"] and p["region"] == {
        "kind": "partition",
        "node": "pr_events",
        "dim": "region",
        "type": "str",
        "implicit": True,
    }


# ------------------------------------------------------------------ sinks


def test_sink_bindings_and_pullable():
    async def send(report: File, date: datetime, to: str = "team@x") -> None:
        raise NotImplementedError

    t = env.task(consumes_artifacts={"report": report})(send)
    lin = extract_task_lineage(t)
    b = _b(t)
    assert b["produces"] == [] and b["pullable"] is True and b["unpullable_reason"] == ""
    assert lin.produces == [] and lin.consumes == ["pr_report"] and lin.warnings() == []
    assert set(b["artifacts"]) == {"pr_report"}

    async def send2(report: File, channel: str) -> None:
        raise NotImplementedError

    unplannable = env.task(consumes_artifacts={"report": report})(send2)
    lin = extract_task_lineage(unplannable)
    assert lin.pullable is False and "parameter 'channel'" in lin.unpullable_reason and lin.warnings() == []

    async def send3(x: File) -> None:
        raise NotImplementedError

    label_only = env.task(consumes_artifacts={"x": artifacts.Artifact("pr_bare")})(send3)
    assert extract_task_lineage(label_only).pullable is False
