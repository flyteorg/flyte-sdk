"""Handles in flyte.remote.Artifact, `flyte deploy --label` and the deploy summary, and flyte.materialize."""

import re
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from flyteidl2.artifact import artifact_service_pb2

import flyte
import flyte.artifacts as artifacts
from flyte.io import DataFrame
from flyte.remote import Artifact
from flyte.remote._artifact import PARTITION_FIELD_PREFIX, TIME_PARTITION_FIELD

features = artifacts.Artifact("features", type=DataFrame, partitions={"date": artifacts.Daily}, project="ml")
events = artifacts.Artifact("events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})


def _client():
    client = MagicMock()
    client.artifact_service.list_artifacts = AsyncMock(
        return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")
    )
    client.artifact_service.get_artifact = AsyncMock(return_value=artifact_service_pb2.GetArtifactResponse())
    return client


def _patched(client):
    cfg = MagicMock(org="o", project="proj", domain="dev")
    return (
        patch("flyte.remote._artifact.ensure_client"),
        patch("flyte.remote._artifact.get_init_config", return_value=cfg),
        patch("flyte.remote._artifact.get_client", return_value=client),
    )


def _by(filters, field):
    return [list(f.values) for f in filters if f.field == field]


# ------------------------------------------------------------------ remote


@pytest.mark.asyncio
async def test_get_with_handle_floors_to_granularity_and_uses_scope():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3, pytest.raises(ValueError, match="No version of artifact 'features'"):
        # A datetime would be an hourly partition by name; through a Daily handle it is that day.
        await Artifact.get.aio(features, date=datetime(2026, 9, 8, 17, 30))
    req = client.artifact_service.list_artifacts.await_args[0][0]
    assert req.name == "features"
    assert req.project_id.name == "ml"  # the handle's project
    assert _by(req.request.filters, TIME_PARTITION_FIELD) == [["2026-09-08T00:00:00Z"]]


@pytest.mark.asyncio
async def test_get_with_handle_no_partitions_uses_get_artifact():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3:
        await Artifact.get.aio(events)
    req = client.artifact_service.get_artifact.await_args[0][0]
    assert req.name.name == "events" and req.name.project == "proj"


@pytest.mark.asyncio
async def test_get_by_name_unchanged():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3, pytest.raises(ValueError):
        await Artifact.get.aio("features", date=datetime(2026, 9, 8, 17, 30))
    req = client.artifact_service.list_artifacts.await_args[0][0]
    assert _by(req.request.filters, TIME_PARTITION_FIELD) == [["2026-09-08T17:00:00Z"]]  # hourly by name


@pytest.mark.asyncio
async def test_listall_with_handle_and_time_range():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3:
        out = [
            a
            async for a in Artifact.listall.aio(
                name=events,
                date=flyte.TimeRange("2026-08-01", "2026-08-31"),
                region=["us", "eu"],
                latest_per_partition=True,
            )
        ]
    assert out == []
    req = client.artifact_service.list_artifacts.await_args[0][0]
    assert req.name == "events" and req.latest_per_partition
    assert _by(req.request.filters, TIME_PARTITION_FIELD) == [["2026-08-01T00:00:00Z"], ["2026-08-31T00:00:00Z"]]
    assert _by(req.request.filters, f"{PARTITION_FIELD_PREFIX}region") == [["us", "eu"]]


@pytest.mark.asyncio
async def test_listall_rejects_trailing_window_through_handle():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3, pytest.raises(ValueError, match="not a trailing window"):
        async for _ in Artifact.listall.aio(events, date=artifacts.TimeRange(days=3)):
            pass


@pytest.mark.asyncio
async def test_handle_rejects_undeclared_dimension():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3, pytest.raises(ValueError, match=r"no partition dimension 'regoin'; declared: date, region"):
        await Artifact.get.aio(events, regoin="us")
    client.artifact_service.list_artifacts.assert_not_awaited()


@pytest.mark.asyncio
async def test_partition_values_accepts_handle():
    client = _client()
    client.artifact_service.list_partition_values = AsyncMock(
        return_value=artifact_service_pb2.ListPartitionValuesResponse(values=["eu", "us"])
    )
    client.artifact_service.get_artifact_schema = AsyncMock(
        return_value=artifact_service_pb2.GetArtifactSchemaResponse()
    )
    p1, p2, p3 = _patched(client)
    with p1, p2, p3:
        assert await Artifact.partition_values.aio(features, "region") == ["eu", "us"]
    assert client.artifact_service.get_artifact_schema.await_args[0][0].name.name == "features"
    req = client.artifact_service.list_partition_values.await_args[0][0]
    assert req.name == "features" and req.project_id.name == "ml"


@pytest.mark.asyncio
async def test_listall_by_name_accepts_absolute_range():
    client = _client()
    p1, p2, p3 = _patched(client)
    with p1, p2, p3:
        async for _ in Artifact.listall.aio("events", date=flyte.TimeRange(date(2026, 8, 1), date(2026, 8, 2))):
            pass
    req = client.artifact_service.list_artifacts.await_args[0][0]
    assert _by(req.request.filters, TIME_PARTITION_FIELD) == [["2026-08-01T00:00:00Z"], ["2026-08-02T00:00:00Z"]]


# ------------------------------------------------------------------ CLI


def _tmp_module(tmp_path):
    src = """
from datetime import datetime
import flyte, flyte.artifacts as artifacts
from flyte.io import File
env = flyte.TaskEnvironment(name="cli_lineage")
snap = artifacts.Artifact("snap", type=File, partitions={"date": artifacts.Daily})
@env.task(produces_artifacts=(snap,), consumes_artifacts={"date": snap.get_partition_value("date")})
async def ok(date: datetime) -> File: ...
@env.task(produces_artifacts=(artifacts.Artifact("snap2", type=File),))
async def snapshot(day: str) -> File: ...
"""
    f = tmp_path / "lineage_mod.py"
    f.write_text(src)
    return f


def test_deploy_label_option_and_summary(tmp_path):
    from flyte.cli.main import main

    f = _tmp_module(tmp_path)
    captured = {}

    def fake_deploy(*envs, **kwargs):
        from flyte._deploy import lineage_summary, plan_deploy

        captured["labels"] = kwargs.get("labels")
        captured["envs"] = [e.name for e in envs]
        summary = lineage_summary(plan_deploy(*envs), labels=kwargs.get("labels"), root_dir=tmp_path)
        return [MagicMock(env_repr=list, table_repr=list, lineage=summary)]

    runner = CliRunner()
    with patch("flyte.deploy", side_effect=fake_deploy), patch("flyte.cli._common.CLIConfig.init"):
        result = runner.invoke(
            main,
            [
                "deploy",
                "--label",
                "team=ml",
                "--label",
                "lineage.consumes=x",
                "--root-dir",
                str(tmp_path),
                str(f),
                "env",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert captured == {"labels": {"team": "ml", "lineage.consumes": "x"}, "envs": ["cli_lineage"]}
    assert "✓ 2 tasks, 2 artifact handles, 0 dependency edges resolved" in result.output
    assert (
        "! cli_lineage.snapshot is not pullable: parameter 'day' has no default, no binding, and is not named like a "
        "partition dimension of snap2. Give it a default, bind it (artifacts.partition(dim) for a partition value), "
        "or mark it artifacts.required() so every materialization must supply it."
    ) in result.output
    assert "Deployed anyway. The task still runs when called directly; it cannot be a materialize target." in (
        result.output
    )


def test_deploy_without_lineage_prints_no_summary(tmp_path):
    """Users who don't declare artifacts see the deploy output they always did."""
    from flyte._deploy import lineage_summary, plan_deploy
    from flyte.cli.main import main

    f = tmp_path / "plain_mod.py"
    f.write_text(
        "import flyte\nenv = flyte.TaskEnvironment(name='cli_plain')\n@env.task\nasync def t(x: int) -> int: ...\n"
    )

    def fake_deploy(*envs, **kwargs):
        return [MagicMock(env_repr=list, table_repr=list, lineage=lineage_summary(plan_deploy(*envs)))]

    with patch("flyte.deploy", side_effect=fake_deploy), patch("flyte.cli._common.CLIConfig.init"):
        result = CliRunner().invoke(main, ["deploy", "--root-dir", str(tmp_path), str(f), "env"])
    assert result.exit_code == 0, result.output
    assert "artifact handles" not in result.output


def test_flyte_deploy_attaches_its_lineage_summary():
    import flyte._deploy as d

    summary = d.lineage_summary([], root_dir=None)
    deployment = d.Deployment(envs={})
    with (
        patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})),
        patch.object(d, "_build_images_for_plans", AsyncMock(return_value=None)),
        patch.object(d, "apply", AsyncMock(return_value=deployment)),
        patch.object(d, "lineage_summary", return_value=summary) as ls,
    ):
        env = flyte.TaskEnvironment(name="attach_summary")
        (out,) = d.deploy(env)
    assert out.lineage is summary
    ls.assert_called_once()


def test_deploy_label_rejects_bad_pair(tmp_path):
    from flyte.cli.main import main

    f = _tmp_module(tmp_path)
    result = CliRunner().invoke(main, ["deploy", "--label", "noequals", str(f), "env"])
    assert result.exit_code != 0
    assert "Expected key-value pair of the form key=value" in result.output


def test_flyte_deploy_validates_before_building_and_threads_labels():
    import flyte._deploy as d

    env = flyte.TaskEnvironment(name="deploy_lineage_check")

    @env.task(consumes_artifacts={"nope": events})
    async def t(x: int) -> None: ...

    with (
        patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})),
        patch.object(d, "_build_images_for_plans", AsyncMock()) as build,
    ):
        with pytest.raises(
            flyte.errors.LineageDeclarationError, match="consumes_artifacts key 'nope' names no parameter"
        ):
            d.deploy(env)
        build.assert_not_called()

    env_ok = flyte.TaskEnvironment(name="deploy_lineage_ok")

    @env_ok.task
    async def u(x: int) -> None: ...

    with (
        patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})),
        patch.object(d, "_build_images_for_plans", AsyncMock(return_value=None)),
        patch.object(d, "apply", AsyncMock(return_value="deployed")) as apply,
    ):
        assert d.deploy(env_ok, labels={"team": "ml"}) == ["deployed"]
    assert apply.await_args.kwargs["labels"] == {"team": "ml"}


def test_flyte_deploy_rejects_deploy_wide_produces():
    import flyte._deploy as d

    env = flyte.TaskEnvironment(name="deploy_produces_label")
    with patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})):
        with pytest.raises(flyte.errors.LineageDeclarationError, match="cannot be a deploy-wide label"):
            d.deploy(env, labels={"lineage.produces": "x"})


def test_flyte_deploy_rejects_reserved_label():
    import flyte._deploy as d

    env = flyte.TaskEnvironment(name="deploy_reserved")

    @env.task
    async def u(x: int) -> None: ...

    with patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})):
        with pytest.raises(
            flyte.errors.LineageDeclarationError, match=re.escape("'lineage.bindings' is in the reserved")
        ):
            d.deploy(env, labels={"lineage.bindings": "{}"})
