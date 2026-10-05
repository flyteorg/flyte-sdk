"""`Artifact.ref`: reading an artifact another codebase owns, without importing its handle."""

import asyncio
import json
import re
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts._lineage import check_conflicts, extract_task_lineage, summarize
from flyte.artifacts._refs import check_references
from flyte.errors import LineageDeclarationError
from flyte.io import DataFrame, File
from flyte.remote._artifact import PartitionSchema

env = flyte.TaskEnvironment(name="analytics-ref")

features = artifacts.Artifact.ref("features", partitions={"date": artifacts.Daily}, project="ml")
churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})
daily_report = artifacts.Artifact("daily_report", type=DataFrame, partitions={"date": artifacts.Daily})


@env.task(
    produces_artifacts=(daily_report,),
    consumes_artifacts={"week": features.window(date=artifacts.TimeRange(days=7)), "model": churn_model},
)
async def report(week: List[DataFrame], model: File, date: datetime) -> DataFrame: ...


def test_a_reference_is_a_handle_that_reads_like_one():
    assert isinstance(features, artifacts.Artifact) and isinstance(features, artifacts.ArtifactRef)
    assert features.reference and not daily_report.reference
    assert repr(features) == "Artifact.ref('features', partitions={date: Daily})"
    assert features.all("date").kind == "all"
    assert flyte.OnArtifact(churn_model).name == "churn_model"  # same project: a trigger can watch it


def test_a_reference_wires_the_same_edges_and_records_only_what_it_states():
    s = summarize([report])
    assert (s.handles, s.edges) == (3, 2)
    record = json.loads(extract_task_lineage(report).labels["lineage.bindings"])["artifacts"]["features"]
    assert record["reference"] is True and record["project"] == "ml"
    assert record["dims"] == [{"name": "date", "kind": "time", "granularity": "day"}]
    assert not {"kind", "description", "source", "identity"} & set(record)


def test_a_reference_cannot_be_produced_or_published():
    with pytest.raises(LineageDeclarationError, match=re.escape("'features', a reference (Artifact.ref)")):

        @env.task(produces_artifacts=(features,))
        async def remake(date: datetime) -> DataFrame: ...

        extract_task_lineage(remake)
    with pytest.raises(ValueError, match="can only be read"):
        features.at(date=datetime(2026, 9, 8))
    with pytest.raises(ValueError, match="can only be read"):
        features.expect(date=["2026-09-08"])


def test_a_reference_must_agree_with_the_owner_handle_in_the_same_deploy():
    owner = artifacts.Artifact("features", type=DataFrame, partitions={"date": artifacts.Daily}, kind="data")
    assert check_conflicts([features, owner])["features"] is owner  # the owner's record, whatever the order
    stale = artifacts.Artifact.ref("features", partitions={"date": artifacts.Daily, "region": str})
    with pytest.raises(LineageDeclarationError, match="different dimensions"):
        check_conflicts([owner, stale])


def _schema(time_key="date", granularity="day", keys=()):
    return PartitionSchema(time_key=time_key, granularity=granularity, keys=tuple(keys), declared=True)


def test_deploy_checks_references_against_the_registry():
    get = AsyncMock(side_effect=[_schema(), _schema()])
    with patch("flyte.remote.Artifact.get_schema") as gs:
        gs.aio = get
        result = asyncio.run(check_references([features, churn_model, daily_report, features]))
    assert result.checked == 2 and not result.notes
    assert {c.args[0] for c in get.call_args_list} == {"features", "churn_model"}  # owners' handles not looked up
    assert get.call_args_list[0].kwargs == {"project": "ml", "domain": None}


def test_a_stale_reference_fails_deploy_with_the_line_to_paste():
    with patch("flyte.remote.Artifact.get_schema") as gs:
        gs.aio = AsyncMock(return_value=_schema(keys=("region",)))
        with pytest.raises(LineageDeclarationError) as e:
            asyncio.run(check_references([features]))
        with pytest.raises(LineageDeclarationError, match=re.escape('Artifact.ref("churn_model", type=File, ')):
            asyncio.run(check_references([churn_model]))
    msg = str(e.value)
    assert "states partitions [date: Daily], but its owner's declaration in the registry has [date: Daily, region]" in (
        msg
    )
    suggested = 'artifacts.Artifact.ref("features", partitions={"date": artifacts.Daily, "region": str}, project="ml")'
    assert f"features = {suggested}" in msg


def test_a_reference_the_registry_does_not_know_yet_is_a_note():
    class NotFound(Exception):
        pass

    with patch("flyte.remote.Artifact.get_schema") as gs:
        gs.aio = AsyncMock(side_effect=NotFound("artifact features not found"))
        result = asyncio.run(check_references([features]))
    assert result.checked == 0
    assert result.notes == [
        "artifact reference 'features' is not in the registry yet (no declaration or version), so its partitions "
        "are unchecked; they are checked on the next deploy after its owner publishes"
    ]


def test_a_slow_registry_lookup_is_a_note_not_a_failure(monkeypatch):
    import flyte.artifacts._refs as refs

    monkeypatch.setattr(refs, "REF_CHECK_TIMEOUT", 0.05)

    async def slow(*a, **k):
        await asyncio.sleep(5)

    with patch("flyte.remote.Artifact.get_schema") as gs:
        gs.aio = slow
        result = asyncio.run(check_references([features]))
    assert result.checked == 0
    assert len(result.notes) == 1 and "timed out" in result.notes[0]


def test_the_overall_reference_check_is_bounded(monkeypatch):
    import flyte.artifacts._refs as refs

    monkeypatch.setattr(refs, "REF_CHECK_TIMEOUT", 10)
    monkeypatch.setattr(refs, "REF_CHECK_TOTAL_TIMEOUT", 0.05)

    async def slow(*a, **k):
        await asyncio.sleep(5)

    with patch("flyte.remote.Artifact.get_schema") as gs:
        gs.aio = slow
        result = asyncio.run(check_references([features, churn_model]))
    assert result.checked == 0
    assert len(result.notes) == 2 and all("timed out" in n for n in result.notes)
