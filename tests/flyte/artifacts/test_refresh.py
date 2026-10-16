"""Keeping an artifact fresh: `Artifact(..., refresh=...)` (the owner) and `handle.materialize_on(...)` (anyone)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts import _refresh
from flyte.artifacts._lineage import BINDINGS_LABEL, CONSUMES_LABEL, extract_task_lineage
from flyte.artifacts._refresh import build_refresh_task, refresh_envs
from flyte.errors import LineageDeclarationError
from flyte.io import File


@pytest.fixture(autouse=True)
def _fresh_envs(monkeypatch):
    # The generated environments are shared per target within a process; isolate each test.
    monkeypatch.setattr(_refresh, "_ENVS", {})
    monkeypatch.setattr(_refresh, "default_image", lambda: "example.com/refresh:dev")
    # A refresh body may set FLYTE_FACTORY_IMAGE in this process; set-then-delete makes monkeypatch restore it.
    monkeypatch.setenv("FLYTE_FACTORY_IMAGE", "unset")
    monkeypatch.delenv("FLYTE_FACTORY_IMAGE")


raw = artifacts.Artifact("rf_raw", type=File, partitions={"date": artifacts.Hourly, "region": str}, source=True)
flat = artifacts.Artifact("rf_flat", type=File)


def _owner(refresh, name="rf_daily", partitions=None):
    return artifacts.Artifact(name, type=File, partitions=partitions or {"date": artifacts.Daily}, refresh=refresh)


def _producing_env(handle):
    env = flyte.TaskEnvironment(name=f"producer-{handle.name.replace('_', '-')}")

    async def make(date: datetime) -> File:
        raise NotImplementedError

    env.task(produces_artifacts=(handle,))(make)
    return env


def _b(task):
    return json.loads(extract_task_lineage(task).labels[BINDINGS_LABEL])


def test_an_owner_policy_is_registered_by_the_deploy_that_produces_it():
    daily = _owner(artifacts.Refresh(flyte.Cron("0 6 * * *"), lag=artifacts.TimeRange(days=1), name="nightly"))
    (renv,) = refresh_envs([_producing_env(daily)])
    assert renv.name == "refresh-rf-daily" and list(renv.tasks) == ["refresh-rf-daily.nightly"]
    task = renv.tasks["refresh-rf-daily.nightly"]
    b = _b(task)
    assert b["materialize_on"] == {
        "target": "rf_daily",
        "event": {"kind": "cron", "cron": "0 6 * * *", "timezone": "UTC"},
        "lag": {"days": 1, "hours": 0},
    }
    assert b["produces"] == [] and b["parameters"] == {} and b["pullable"] is False
    assert extract_task_lineage(task).labels[CONSUMES_LABEL] == "rf_daily"
    assert b["src_file"].endswith("test_refresh.py")  # where the policy was declared, not the SDK
    (trig,) = task.triggers
    assert trig.name == "nightly" and trig.inputs == {"trigger_time": flyte.TriggerTime}
    assert renv.image == "example.com/refresh:dev"


def test_only_a_producing_deploy_registers_the_policy():
    daily = _owner(flyte.Cron("0 6 * * *"))
    reader = flyte.TaskEnvironment(name="reader")

    async def read(x: File, date: datetime) -> str:
        raise NotImplementedError

    reader.task(consumes_artifacts={"x": daily})(read)
    assert refresh_envs([reader]) == []
    assert artifacts.Artifact.ref("rf_daily", partitions={"date": artifacts.Daily}).refresh == ()


def test_bare_events_and_lists_are_policies():
    daily = _owner([flyte.FixedRate(60), raw])
    assert [type(p) for p in daily.refresh] == [artifacts.Refresh, artifacts.Refresh]
    (renv,) = refresh_envs([_producing_env(daily)])
    assert list(renv.tasks) == ["refresh-rf-daily.rf_daily_on_schedule", "refresh-rf-daily.rf_daily_on_rf_raw"]
    assert _b(renv.tasks["refresh-rf-daily.rf_daily_on_schedule"])["materialize_on"]["event"] == {
        "kind": "fixed_rate",
        "interval_minutes": 60,
    }


def test_a_source_policy_passes_the_partition_of_the_new_version():
    regional = _owner(
        artifacts.Refresh(raw, region="us"), name="rf_regional", partitions={"date": artifacts.Daily, "region": str}
    )
    (renv,) = refresh_envs([_producing_env(regional)])
    task = renv.tasks["refresh-rf-regional.rf_regional_on_rf_raw"]
    b = _b(task)
    assert b["materialize_on"] == {
        "target": "rf_regional",
        "event": {"kind": "source", "source": "rf_raw", "filter": {"region": "us"}},
        "lag": None,
    }
    assert set(b["artifacts"]) == {"rf_regional", "rf_raw"}
    (trig,) = task.triggers
    assert trig.automation.name == "rf_raw" and trig.automation.partitions == {"region": "us"}
    assert trig.inputs == {"date": flyte.TriggeredPartition("date"), "region": flyte.TriggeredPartition("region")}
    assert list(task.native_interface.inputs) == ["date", "region"]


def test_a_reader_keeps_a_reference_fresh():
    features = artifacts.Artifact.ref("rf_features", partitions={"date": artifacts.Daily}, project="ml")
    env = features.materialize_on(flyte.Cron("0 * * * *"), lag=artifacts.TimeRange(hours=1), image="img:1")
    assert isinstance(env, flyte.TaskEnvironment) and env.name == "refresh-rf-features-ml" and env.image == "img:1"
    again = features.materialize_on(raw)
    assert again is env
    assert list(env.tasks) == [
        "refresh-rf-features-ml.rf_features_on_schedule",
        "refresh-rf-features-ml.rf_features_on_rf_raw",
    ]
    record = _b(env.tasks["refresh-rf-features-ml.rf_features_on_schedule"])["artifacts"]["rf_features"]
    assert record["reference"] is True and record["project"] == "ml"


def test_two_different_policies_with_one_name_fail():
    daily = _owner(None)
    daily.materialize_on(flyte.Cron("0 6 * * *"))
    daily.materialize_on(flyte.Cron("0 6 * * *"))  # the same policy again: idempotent
    with pytest.raises(LineageDeclarationError, match="give one a name="):
        daily.materialize_on(flyte.Cron("0 7 * * *"))


def test_the_task_is_rebuilt_at_run_time_from_its_spec():
    daily = _owner(artifacts.Refresh(flyte.Cron("0 6 * * *"), lag=artifacts.TimeRange(days=1)))
    (renv,) = refresh_envs([_producing_env(daily)])
    task = renv.tasks["refresh-rf-daily.rf_daily_on_schedule"]
    args = task.task_resolver.loader_args(task)
    assert args[:2] == ["task_builder", "flyte.artifacts._refresh.build_refresh_task"] and args[2] == "spec"
    rebuilt = build_refresh_task(args[3])
    assert rebuilt.name == task.name and list(rebuilt.native_interface.inputs) == ["trigger_time"]
    assert rebuilt.triggers == ()


class _StubMaterialize:
    def __init__(self):
        self.calls = []

    async def aio(self, target, **kwargs):
        self.calls.append((target, kwargs))
        return "run"


@pytest.mark.asyncio
async def test_bodies_materialize_the_partition_the_trigger_names(monkeypatch):
    stub = _StubMaterialize()
    monkeypatch.setattr(flyte, "materialize", stub)
    daily = _owner(artifacts.Refresh(flyte.Cron("0 6 * * *"), lag=artifacts.TimeRange(days=1)))
    regional = _owner(raw, name="rf_regional", partitions={"date": artifacts.Daily, "region": str})
    (d_env,) = refresh_envs([_producing_env(daily)])
    (r_env,) = refresh_envs([_producing_env(regional)])
    on_schedule = d_env.tasks["refresh-rf-daily.rf_daily_on_schedule"]
    await on_schedule.func(trigger_time=datetime(2026, 9, 9, 6, tzinfo=timezone.utc))
    await r_env.tasks["refresh-rf-regional.rf_regional_on_rf_raw"].func(
        date=datetime(2026, 9, 8, 17, tzinfo=timezone.utc), region="us"
    )
    (t1, k1), (t2, k2) = stub.calls
    assert t1 is daily and k1 == {
        "partitions": {"date": datetime(2026, 9, 8, tzinfo=timezone.utc)},
        "project": None,
        "domain": None,
    }
    assert t2 is regional
    assert k2["partitions"] == {"date": datetime(2026, 9, 8, tzinfo=timezone.utc), "region": "us"}


@pytest.mark.parametrize(
    "target,policy,err,match",
    [
        (flat, flyte.Cron("0 6 * * *"), ValueError, "has no partitions"),
        ({"region": str}, flyte.FixedRate(5), ValueError, "no time dimension"),
        (None, artifacts.Refresh(raw, lag=artifacts.TimeRange(days=1)), ValueError, "lag applies to a schedule"),
        ({"date": artifacts.Daily, "shard": str}, raw, ValueError, "dimension\\(s\\) 'shard'"),
        (None, "nightly", TypeError, "the event must be"),
        (None, artifacts.Refresh(flyte.Cron("0 6 * * *"), region="us"), ValueError, "source event only"),
        (None, artifacts.Refresh(raw, nope="x"), ValueError, "no partition dimension 'nope'"),
        (
            None,
            artifacts.Refresh(flyte.Cron("0 6 * * *"), lag=artifacts.TimeRange("2026-01-01", "2026-01-02")),
            ValueError,
            "trailing TimeRange",
        ),
    ],
)
def test_a_policy_that_cannot_work_fails_at_the_declaration(target, policy, err, match):
    with pytest.raises(err, match=match):
        if isinstance(target, artifacts.Artifact):
            target.materialize_on(policy)
        else:
            _owner(policy, partitions=target)


def test_the_derived_factory_reuses_the_refresh_task_image(monkeypatch):
    from types import SimpleNamespace

    from flyte._internal.image_cache import ImageCache

    monkeypatch.delenv("FLYTE_FACTORY_IMAGE", raising=False)
    cache = ImageCache(image_lookup={"refresh-rf-daily": "registry/factory:abc"})
    monkeypatch.setattr(flyte, "ctx", lambda: SimpleNamespace(compiled_image_cache=cache))
    _refresh._reuse_own_image("rf_daily")
    import os

    assert os.environ["FLYTE_FACTORY_IMAGE"] == "registry/factory:abc"
    monkeypatch.setenv("FLYTE_FACTORY_IMAGE", "pinned:1")
    _refresh._reuse_own_image("rf_daily")
    assert os.environ["FLYTE_FACTORY_IMAGE"] == "pinned:1"  # an explicit pin wins


def test_refresh_envs_scan_depends_on_and_are_described():
    from flyte.artifacts._lineage import LineageSummary
    from flyte.artifacts._refresh import describe_refresh_envs

    daily = _owner(artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=artifacts.TimeRange(days=1), name="nightly"))
    producer = _producing_env(daily)
    top = flyte.TaskEnvironment(name="rf-top", depends_on=[producer])
    (renv,) = refresh_envs([top])
    lines = describe_refresh_envs([renv])
    assert lines == ["+ refresh-rf-daily: nightly (cron 0 2 * * *)"]
    summary = LineageSummary(refreshes=lines)
    assert summary.relevant
    assert "+ refresh-rf-daily: nightly (cron 0 2 * * *)" in summary.render()


def test_schedule_floors_in_the_cron_timezone():
    from datetime import timedelta

    # 22:00 in Los Angeles on Sept 8 is 05:00 UTC on Sept 9; the partition is the local day.
    fire = datetime(2026, 9, 9, 5, tzinfo=timezone.utc)
    t = _refresh.schedule_partition_time(fire, "America/Los_Angeles", timedelta(0))
    assert artifacts.Daily.floor(t) == datetime(2026, 9, 8, tzinfo=timezone.utc)
    assert artifacts.Daily.floor(_refresh.schedule_partition_time(fire, "UTC", timedelta(0))) == datetime(
        2026, 9, 9, tzinfo=timezone.utc
    )


def test_targets_that_slug_alike_fail():
    artifacts.Artifact("rf_same", partitions={"date": artifacts.Daily}).materialize_on(flyte.Cron("0 6 * * *"))
    with pytest.raises(LineageDeclarationError, match="rename one"):
        artifacts.Artifact("rf-same", partitions={"date": artifacts.Daily}).materialize_on(flyte.Cron("0 6 * * *"))


def test_a_later_policy_cannot_change_the_image_or_resources():
    h = artifacts.Artifact("rf_img", partitions={"date": artifacts.Daily})
    env = h.materialize_on(flyte.Cron("0 6 * * *"), image="example.com/a:1")
    assert env.image == "example.com/a:1"
    assert h.materialize_on(flyte.Cron("0 7 * * *"), name="later") is env  # no image: reuses
    with pytest.raises(LineageDeclarationError, match="Use the same image"):
        h.materialize_on(flyte.Cron("0 8 * * *"), name="other", image="example.com/b:2")
    with pytest.raises(LineageDeclarationError, match="resources"):
        h.materialize_on(flyte.Cron("0 8 * * *"), name="big", resources=flyte.Resources(cpu="4"))


def test_resources_and_keyword_named_filters():
    src = artifacts.Artifact("rf_src", partitions={"date": artifacts.Daily, "name": str})
    tgt = artifacts.Artifact("rf_tgt", partitions={"date": artifacts.Daily, "name": str})
    env = tgt.materialize_on(src, partitions={"name": "alpha"}, resources=flyte.Resources(cpu="2", memory="2Gi"))
    assert env.resources == flyte.Resources(cpu="2", memory="2Gi")
    (task,) = env.tasks.values()
    assert task.triggers[0].automation.partitions == {"name": "alpha"}
    with pytest.raises(ValueError, match="given twice"):
        artifacts.Refresh(src, partitions={"x": "a"}, x="b")


def test_same_target_name_in_two_projects_gets_two_environments():
    own = artifacts.Artifact("rf_scoped", partitions={"date": artifacts.Daily})
    theirs = artifacts.Artifact.ref("rf_scoped", partitions={"date": artifacts.Daily}, project="ml", domain="prod")
    a = own.materialize_on(flyte.Cron("0 6 * * *"))
    b = theirs.materialize_on(flyte.Cron("0 6 * * *"))
    assert (a.name, b.name) == ("refresh-rf-scoped", "refresh-rf-scoped-ml-prod")
    # A name that still collides after qualification fails, whatever the scope.
    with pytest.raises(LineageDeclarationError, match="rename one"):
        artifacts.Artifact("rf_scoped-ml-prod", partitions={"date": artifacts.Daily}).materialize_on(
            flyte.Cron("0 6 * * *")
        )


def test_generated_trigger_names_are_dns_labels():
    from flyte.artifacts._refresh import _trigger_name

    assert _trigger_name("keep_report_fresh") == "keep-report-fresh"
    assert _trigger_name("_" * 10) == "trigger"
    long = _trigger_name("a" * 62 + "_b" * 10)
    assert len(long) <= 63 and not long.endswith("-")
    h = artifacts.Artifact("rf_" + "x" * 70, partitions={"date": artifacts.Daily})
    env = h.materialize_on(flyte.Cron("0 6 * * *"))
    (task,) = env.tasks.values()
    (trigger,) = task.triggers
    assert len(trigger.name) <= 63 and re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", trigger.name)


def test_refresh_name_follows_the_lineage_identifier_rule():
    h = artifacts.Artifact("rf_ident", partitions={"date": artifacts.Daily})
    with pytest.raises(ValueError, match="valid identifier"):
        h.materialize_on(flyte.Cron("0 6 * * *"), name="nächtlich")
    with pytest.raises(ValueError, match="valid identifier"):
        h.materialize_on(flyte.Cron("0 6 * * *"), name="n" * 65)
