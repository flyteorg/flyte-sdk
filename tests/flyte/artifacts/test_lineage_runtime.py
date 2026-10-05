"""Run-time publishing of declared handles: `produces_artifacts=(handle, ...)`."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional, Tuple

import pytest
from flyteidl2.core import artifact_id_pb2

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.runtime.convert import (
    convert_from_native_to_inputs,
    convert_from_native_to_outputs,
)
from flyte._internal.runtime.taskrunner import _handle_declarations
from flyte.artifacts._lineage import declared_output_metadata
from flyte.errors import RuntimeUserError
from flyte.io import File
from flyte.models import ActionID, RawDataPath, TaskContext

from . import proposal_tasks as P

UTC = timezone.utc
env = flyte.TaskEnvironment(name="lineage-runtime")

model = artifacts.Artifact("model_rt", type=File, partitions={"date": artifacts.Daily}, kind="model", description="m")
pair_a = artifacts.Artifact("pair_a", type=File, partitions={"date": artifacts.Daily, "region": str})
pair_b = artifacts.Artifact("pair_b", type=File, identity="content")


@env.task(produces_artifacts=(model,), consumes_artifacts={"as_of": model.get_partition_value("date")})
async def fit(as_of: datetime) -> File:
    return File(path="s3://bucket/model.pt")


@env.task(produces_artifacts=(model,), consumes_artifacts={"as_of": model.get_partition_value("date")})
async def fit_with_card(as_of: datetime) -> File:
    card = artifacts.Card(uri="s3://bucket/card.html", format="html", card_type="model")
    return artifacts.new(File(path="s3://bucket/model.pt"), model.at(date=date(2026, 1, 1), card=card, version="v9"))


@env.task(
    produces_artifacts=(pair_a, pair_b),
    consumes_artifacts={"day": pair_a.get_partition_value("date"), "region": pair_a.get_partition_value("region")},
)
async def two(day: date, region: str) -> Tuple[File, File]:
    return File(path="s3://bucket/a"), File(path="s3://bucket/b")


@env.task(produces_artifacts=(model,), consumes_artifacts={"as_of": model.get_partition_value("date")})
async def returns_int(as_of: datetime) -> int:
    return 3


def _time(pa) -> datetime:
    return pa.time_partition.value.time_value.ToDatetime(tzinfo=UTC)


def test_declared_output_metadata_partitions_from_get_partition_value():
    decl = declared_output_metadata(fit, {"as_of": datetime(2026, 9, 8, 17)})
    assert set(decl) == {"o0"} and decl["o0"].handle is model and decl["o0"].error is None
    md = {k: v.metadata for k, v in decl.items()}
    assert md["o0"].name == "model_rt" and md["o0"].kind == "model" and md["o0"].description == "m"
    assert md["o0"].partitions == {"date": artifacts.TimePartition(datetime(2026, 9, 8, tzinfo=UTC), "day")}


def test_partition_from_another_handle_with_same_dim():
    # clean binds date/region against raw_events; they fill `events`' coordinates.
    decl = declared_output_metadata(
        P.clean, {"raw": None, "date": datetime(2026, 9, 8), "region": "us", "min_quality": 3}
    )
    assert decl["o0"].metadata.name == "events"
    assert decl["o0"].metadata.partitions == {
        "date": artifacts.TimePartition(datetime(2026, 9, 8, tzinfo=UTC), "day"),
        "region": "us",
    }


def test_missing_coordinate_is_skipped_and_untyped_tasks_publish_nothing():
    decl = declared_output_metadata(P.clean, {"date": datetime(2026, 9, 8)})
    assert decl["o0"].metadata is None and decl["o0"].error is None
    assert (
        "ingest.clean: not publishing output o0 as 'events': no value for dimension 'region'" in decl["o0"].skip_reason
    )

    @env.task(produces_artifacts=True)
    async def bare() -> File:
        raise NotImplementedError

    assert declared_output_metadata(bare, {}) == {}
    assert _handle_declarations(bare, {}) is None
    assert _handle_declarations(object(), {}) is None  # type: ignore[arg-type]


def test_caller_declared_slots_parse_nothing():
    (decl,) = declared_output_metadata(fit, {"as_of": "garbage"}, skip=["o0"]).values()
    assert decl.caller_declared and decl.metadata is None and decl.error is None and decl.skip_reason is None


@pytest.mark.asyncio
async def test_handle_published_from_declaration():
    out = await convert_from_native_to_outputs(
        await fit.func(as_of=datetime(2026, 9, 8)),
        fit.native_interface,
        fit.name,
        handle_declared=_handle_declarations(fit, {"as_of": datetime(2026, 9, 8)}),
    )
    (pa,) = out.proto_outputs.produced_artifacts
    assert (pa.output, pa.name, pa.version) == ("o0", "model_rt", "")
    assert pa.time_partition.key == "date"
    assert pa.time_partition.granularity == artifact_id_pb2.Granularity.Value("DAY")
    assert _time(pa) == datetime(2026, 9, 8, tzinfo=UTC)
    assert pa.info.user_metadata["flyte.io/kind"] == "model"


@pytest.mark.asyncio
async def test_artifacts_new_metadata_wins_over_declaration():
    out = await convert_from_native_to_outputs(
        await fit_with_card.func(as_of=datetime(2026, 9, 8)),
        fit_with_card.native_interface,
        fit_with_card.name,
        handle_declared=_handle_declarations(fit_with_card, {"as_of": datetime(2026, 9, 8)}),
    )
    (pa,) = out.proto_outputs.produced_artifacts
    assert pa.version == "v9"
    assert _time(pa) == datetime(2026, 1, 1, tzinfo=UTC)
    assert pa.info.card.uri == "s3://bucket/card.html"


@pytest.mark.asyncio
async def test_tuple_outputs_published_by_position():
    vals = await two.func(day=date(2026, 9, 8), region="eu")
    out = await convert_from_native_to_outputs(
        vals,
        two.native_interface,
        two.name,
        handle_declared=_handle_declarations(two, {"day": date(2026, 9, 8), "region": "eu"}),
    )
    a, b = out.proto_outputs.produced_artifacts
    assert (a.output, a.name, b.output, b.name) == ("o0", "pair_a", "o1", "pair_b")
    assert a.partitions.value["region"].static_value == "eu" and _time(a) == datetime(2026, 9, 8, tzinfo=UTC)
    assert not b.HasField("time_partition") and not b.partitions.value


def _task_context(custom_context=None) -> TaskContext:
    return TaskContext(
        action=ActionID(name="a0", run_name="r1"),
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/out",
        run_base_dir="s3://bucket/base",
        report=flyte.report.Report(name="a0"),
        custom_context=dict(custom_context or {}),
    )


@pytest.mark.asyncio
async def test_caller_declarations_win_and_do_not_double_publish():
    """The factory plugin runs tasks inside `artifacts.produces(...)`: one ProducedArtifact per slot."""
    ctx = internal_ctx()
    with ctx.replace_task_context(_task_context()):
        with artifacts.produces(o0=artifacts.Metadata(name="model_rt", partitions={"date": date(2026, 2, 2)})):
            inputs = await convert_from_native_to_inputs(fit.native_interface, as_of=datetime(2026, 9, 8))
    out = await convert_from_native_to_outputs(
        File(path="s3://bucket/m"),
        fit.native_interface,
        fit.name,
        declared=inputs.declared_artifacts,
        handle_declared=_handle_declarations(fit, {"as_of": datetime(2026, 9, 8)}),
    )
    (pa,) = out.proto_outputs.produced_artifacts
    assert _time(pa) == datetime(2026, 2, 2, tzinfo=UTC)  # the caller's identity wins


@pytest.mark.asyncio
async def test_caller_declared_slot_still_carries_handle_description_and_kind():
    """A direct run and the same task under a factory publish the same description and kind."""
    hd = {"as_of": datetime(2026, 9, 8)}
    direct = await convert_from_native_to_outputs(
        File(path="s3://b/m"), fit.native_interface, fit.name, handle_declared=_handle_declarations(fit, hd)
    )
    ctx = internal_ctx()
    with ctx.replace_task_context(_task_context()):
        with artifacts.produces(o0=artifacts.Metadata(name="model_rt", partitions={"date": date(2026, 2, 2)})):
            inputs = await convert_from_native_to_inputs(fit.native_interface, as_of=datetime(2026, 9, 8))
    declared = inputs.declared_artifacts
    factory = await convert_from_native_to_outputs(
        File(path="s3://b/m"),
        fit.native_interface,
        fit.name,
        declared=declared,
        handle_declared=_handle_declarations(fit, hd, skip=declared),
    )
    (d,), (f,) = direct.proto_outputs.produced_artifacts, factory.proto_outputs.produced_artifacts
    assert d.info.description == f.info.description == "m"
    assert d.info.user_metadata["flyte.io/kind"] == f.info.user_metadata["flyte.io/kind"] == "model"

    # The caller's explicit values still win.
    with ctx.replace_task_context(_task_context()):
        explicit = artifacts.Metadata(name="model_rt", description="caller", kind="data")
        with artifacts.produces(o0=explicit):
            inputs = await convert_from_native_to_inputs(fit.native_interface, as_of=datetime(2026, 9, 8))
    declared = inputs.declared_artifacts
    out = await convert_from_native_to_outputs(
        File(path="s3://b/m"),
        fit.native_interface,
        fit.name,
        declared=declared,
        handle_declared=_handle_declarations(fit, hd, skip=declared),
    )
    (pa,) = out.proto_outputs.produced_artifacts
    assert pa.info.description == "caller" and pa.info.user_metadata["flyte.io/kind"] == "data"


@pytest.mark.asyncio
async def test_caller_declaration_survives_unparseable_handle_coordinate():
    """The factory driver path: handle-side parsing can never break a caller declaration."""
    ctx = internal_ctx()
    with ctx.replace_task_context(_task_context()):
        with artifacts.produces(o0=artifacts.Metadata(name="model_rt", partitions={"date": date(2026, 2, 2)})):
            inputs = await convert_from_native_to_inputs(fit.native_interface, as_of=datetime(2026, 9, 8))
    declared = inputs.declared_artifacts
    out = await convert_from_native_to_outputs(
        File(path="s3://bucket/m"),
        fit.native_interface,
        fit.name,
        declared=declared,
        handle_declared=_handle_declarations(fit, {"as_of": "not a date"}, skip=declared),
    )
    assert [pa.name for pa in out.proto_outputs.produced_artifacts] == ["model_rt"]


@pytest.mark.asyncio
async def test_unparseable_coordinate_is_a_clear_user_error(monkeypatch):
    monkeypatch.setenv("FLYTE_LINEAGE_STRICT", "1")
    with pytest.raises(RuntimeUserError) as exc:
        await convert_from_native_to_outputs(
            File(path="s3://b/m"),
            fit.native_interface,
            fit.name,
            handle_declared=_handle_declarations(fit, {"as_of": "not a date"}),
        )
    assert exc.value.code == "BadPartitionValue"
    assert "lineage-runtime.fit: parameter 'as_of' = 'not a date' is not a valid value for dimension 'date'" in str(
        exc.value
    )


opt_handle = artifacts.Artifact("opt_rt", type=File, partitions={"date": artifacts.Daily, "region": str})


@env.task(
    produces_artifacts=(opt_handle,),
    consumes_artifacts={
        "day": opt_handle.get_partition_value("date"),
        "region": opt_handle.get_partition_value("region"),
    },
)
async def optional_region(day: date, region: Optional[str] = None) -> File:
    return File(path="s3://b/x")


@pytest.mark.asyncio
async def test_none_coordinate_skips_the_slot_with_a_warning():
    """The task still runs when called directly: an incomplete partition is not published, and not an error."""
    from unittest.mock import patch

    from flyte._logging import logger

    with patch.object(logger, "warning") as warn:
        out = await convert_from_native_to_outputs(
            File(path="s3://b/x"),
            optional_region.native_interface,
            optional_region.name,
            handle_declared=_handle_declarations(optional_region, {"day": date(2026, 9, 8), "region": None}),
        )
    assert list(out.proto_outputs.produced_artifacts) == []
    assert out.proto_outputs.literals  # the value itself is still returned
    assert (
        "lineage-runtime.optional_region: not publishing output o0 as 'opt_rt': no value for dimension 'region' "
        "(parameter 'region' is None)"
    ) in warn.call_args[0][0]


@pytest.mark.asyncio
async def test_unbound_dimension_skips_but_caller_declaration_publishes():
    unbound = artifacts.Artifact("ub_rt", type=File, partitions={"date": artifacts.Daily, "region": str})

    @env.task(produces_artifacts=(unbound,), consumes_artifacts={"day": unbound.get_partition_value("date")})
    async def partial(day: date) -> File:
        return File(path="s3://b/x")

    hd = {"day": date(2026, 9, 8)}
    direct = await convert_from_native_to_outputs(
        File(path="s3://b/x"), partial.native_interface, partial.name, handle_declared=_handle_declarations(partial, hd)
    )
    assert list(direct.proto_outputs.produced_artifacts) == []
    ctx = internal_ctx()
    with ctx.replace_task_context(_task_context()):
        md = artifacts.Metadata(name="ub_rt", partitions={"date": date(2026, 9, 8), "region": "us"})
        with artifacts.produces(o0=md):
            inputs = await convert_from_native_to_inputs(partial.native_interface, day=date(2026, 9, 8))
    declared = inputs.declared_artifacts
    factory = await convert_from_native_to_outputs(
        File(path="s3://b/x"),
        partial.native_interface,
        partial.name,
        declared=declared,
        handle_declared=_handle_declarations(partial, hd, skip=declared),
    )
    (pa,) = factory.proto_outputs.produced_artifacts
    assert pa.partitions.value["region"].static_value == "us"


four_model = artifacts.Artifact("pl_model", type=File, partitions={"date": artifacts.Daily})
four_metrics = artifacts.Artifact("pl_metrics", type=File, partitions={"date": artifacts.Daily})


@env.task(
    produces_artifacts=(None, four_model, four_metrics, None),
    consumes_artifacts={"day": four_model.get_partition_value("date")},
)
async def four(day: date) -> Tuple[int, File, File, str]:
    return 1, File(path="s3://b/m"), File(path="s3://b/x"), "done"


@pytest.mark.asyncio
async def test_none_placeholders_publish_only_declared_positions():
    hd = {"day": date(2026, 9, 8)}
    decls = _handle_declarations(four, hd)
    assert set(decls) == {"o1", "o2"}
    out = await convert_from_native_to_outputs(
        await four.func(day=date(2026, 9, 8)), four.native_interface, four.name, handle_declared=decls
    )
    assert [(pa.output, pa.name) for pa in out.proto_outputs.produced_artifacts] == [
        ("o1", "pl_model"),
        ("o2", "pl_metrics"),
    ]


@pytest.mark.asyncio
async def test_artifacts_new_missing_a_declared_dimension_is_rejected(monkeypatch):
    monkeypatch.setenv("FLYTE_LINEAGE_STRICT", "1")
    md = opt_handle.at(date=date(2026, 9, 8))  # region left off
    with pytest.raises(RuntimeUserError) as exc:
        await convert_from_native_to_outputs(
            artifacts.new(File(path="s3://b/x"), md),
            optional_region.native_interface,
            optional_region.name,
            handle_declared=_handle_declarations(optional_region, {"day": date(2026, 9, 8), "region": None}),
        )
    assert exc.value.code == "MissingPartition"
    assert "published as 'opt_rt' without partition 'region'" in str(exc.value)


@pytest.mark.asyncio
async def test_artifacts_new_supplies_a_coordinate_the_declaration_lacks():
    md = opt_handle.at(date=date(2026, 9, 8), region="us")
    out = await convert_from_native_to_outputs(
        artifacts.new(File(path="s3://b/x"), md),
        optional_region.native_interface,
        optional_region.name,
        handle_declared=_handle_declarations(optional_region, {"day": date(2026, 9, 8), "region": None}),
    )
    (pa,) = out.proto_outputs.produced_artifacts
    assert pa.partitions.value["region"].static_value == "us"


@pytest.mark.asyncio
async def test_produces_true_unchanged():
    @env.task(produces_artifacts=True)
    async def plain() -> File:
        return File(path="s3://bucket/x")

    out = await convert_from_native_to_outputs(File(path="s3://bucket/x"), plain.native_interface, plain.name)
    assert list(out.proto_outputs.produced_artifacts) == []


@pytest.mark.asyncio
async def test_declared_slot_must_be_artifactable(monkeypatch):
    monkeypatch.setenv("FLYTE_LINEAGE_STRICT", "1")
    with pytest.raises(RuntimeUserError, match="declares output o0 as artifact 'model_rt', but values of type 'int'"):
        await convert_from_native_to_outputs(
            3,
            returns_int.native_interface,
            returns_int.name,
            handle_declared=_handle_declarations(returns_int, {"as_of": datetime(2026, 9, 8)}),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "task", "md", "hd"),
    [
        (File(path="s3://b/m"), fit, None, {"as_of": "not a date"}),  # BadPartitionValue
        (3, returns_int, None, {"as_of": datetime(2026, 9, 8)}),  # NotAnArtifact
        (
            File(path="s3://b/x"),
            optional_region,
            opt_handle.at(date=date(2026, 9, 8)),
            {"day": date(2026, 9, 8), "region": None},
        ),  # MissingPartition
    ],
)
async def test_runtime_lineage_problems_fail_open(monkeypatch, value, task, md, hd):
    """Without FLYTE_LINEAGE_STRICT, a lineage problem publishes nothing for the slot; the output literal stays."""
    monkeypatch.delenv("FLYTE_LINEAGE_STRICT", raising=False)
    v = artifacts.new(value, md) if md is not None else value
    out = await convert_from_native_to_outputs(
        v, task.native_interface, task.name, handle_declared=_handle_declarations(task, hd)
    )
    assert list(out.proto_outputs.produced_artifacts) == []
    assert [nl.name for nl in out.proto_outputs.literals] == list(task.native_interface.outputs)


def test_handle_declarations_fail_open(monkeypatch):
    import flyte.artifacts._lineage as lineage

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(lineage, "declared_output_metadata", boom)
    monkeypatch.delenv("FLYTE_LINEAGE_STRICT", raising=False)
    assert _handle_declarations(fit, {"as_of": datetime(2026, 9, 8)}) is None
    monkeypatch.setenv("FLYTE_LINEAGE_STRICT", "1")
    with pytest.raises(ValueError, match="boom"):
        _handle_declarations(fit, {"as_of": datetime(2026, 9, 8)})
