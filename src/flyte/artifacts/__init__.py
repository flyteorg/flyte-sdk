"""
Artifacts module

This module provides a wrapper method to mark certain outputs as artifacts with associated metadata.
Artifacts are offloaded assets: a flyte.io File, Dir, or DataFrame.

Usage example:
```python
import flyte.artifacts as artifacts
from flyte.io import File

@env.task
async def my_task() -> File:
    file = await File.from_local("weights.pt")
    metadata = artifacts.Metadata(name="my_artifact", version="1.0", description="An example artifact")
    return artifacts.new(file, metadata)
```

Launching with known artifacts:
```python
flyte.run(main, x=flyte.remote.Artifact.get("name", version="1.0"))
```

Retrieve a set of artifacts and pass them as a list
```python
from flyte.remote import Artifact
flyte.run(main, x=[Artifact.get("name1", version="1.0"), Artifact.get("name2", version="2.0")])
```
OR, listing versions of one artifact. `listall` is an iterator, so materialize it
before binding it as an input — a run input must be an `Artifact` or a list of them.
```python
from flyte.remote import Artifact
flyte.run(main, x=list(Artifact.listall(name="name1", limit=5)))
```
Use `Artifact.list_names(search=...)` to browse distinct artifact names instead.

Publishing a model:
```python
metadata = artifacts.Metadata(name="sentiment-model", kind="model")
return artifacts.new(file, metadata)
```
`Metadata.create_model_metadata(...)` sets `kind="model"` for you, alongside the
model-specific attrs (framework, architecture, and so on).

Read it back with `flyte.remote.Artifact.kind`, which returns "model", "data", or
"generic" -- never None. It is stored under a reserved `flyte.io/kind` attr, but
callers should use the property rather than reading `user_metadata` directly, so the
key can move to a typed field later without breaking them.

`kind` is what an artifact *is*; a card's `card_type` is how its card *renders*. An
artifact can have one without the other.

Partitions are part of an artifact's identity. Give a version its partition values
in `Metadata.partitions`; a `date` is a daily time partition, a `datetime` an hourly
one, and anything else is a string partition:
```python
metadata = artifacts.Metadata(name="raw_events", partitions={"date": day, "region": region})
return artifacts.new(file, metadata)
```
Read a partition back with `Artifact.get("raw_events", date=day, region="us")`, list a
range with `Artifact.listall("raw_events", date=(start, end), latest_per_partition=True)`,
and list the values of one key with `Artifact.partition_values("raw_events", "region")`.

Declaring artifacts beside the code that owns them. A handle names an artifact, its type and its
partition dimensions; tasks name handles in their decorator, so deploy knows what each task produces and
consumes before it runs, and the lineage graph emerges from those declarations:
```python
events = artifacts.Artifact("events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
features = artifacts.Artifact("features", type=DataFrame, partitions={"date": artifacts.Daily})

@env.task(
    consumes_artifacts={"per_region": events.all("region"), "date": features.get_partition_value("date")},
    produces_artifacts=(features,),
)
async def featurize(per_region: list[DataFrame], date: datetime) -> DataFrame: ...
```
An artifact owned by code you cannot import (another team's repo) is read through a reference that states its
name and partitions; deploy checks them against the registry (that check is experimental, and needs the Union
lineage service; elsewhere it is skipped with a note):
```python
features = artifacts.Artifact.ref("features", type=DataFrame, partitions={"date": artifacts.Daily}, project="ml")
```

A parameter with no default and no binding that is named like a dimension of what the task produces (`date`
above) carries that dimension's value without a binding; `artifacts.partition("date")` binds a parameter of
any name to it, and `artifacts.required()` marks one every materialization must supply:
```python
@env.task(consumes_artifacts={"day": artifacts.partition("date"), "seed": artifacts.required()},
          produces_artifacts=(model,))
async def train(day: datetime, seed: int) -> File: ...
```

Experimental; requires the Union lineage service and flyteplugins-union: refresh policies (`Refresh`,
`Artifact(refresh=...)`, `handle.materialize_on(...)`) and `flyte.materialize`, which plan and launch work through
the lineage planner in flyteplugins-union.

A task that consumes artifacts and produces none is a sink (a report sent, a dashboard refreshed); it is planned
like any other step. To keep an artifact fresh (materialize it on a schedule or on each new source version), its
owner declares a refresh policy on the handle, and a deploy that produces it registers the trigger; anyone else,
including through an `Artifact.ref`, uses `handle.materialize_on(...)`, which returns an environment to deploy:
```python
daily_report = artifacts.Artifact("daily_report", type=File, partitions={"date": artifacts.Daily},
                                  refresh=artifacts.Refresh(flyte.Cron("0 6 * * *"), lag=artifacts.TimeRange(days=1)))
keep_features_fresh = features.materialize_on(flyte.Cron("0 * * * *"), lag=artifacts.TimeRange(hours=1))
```

Producing artifacts from a task that does not wrap its outputs: the caller declares them.
```python
with artifacts.produces(o0=artifacts.Metadata(name="events", partitions={"date": day})):
    await clean.override(produces_artifacts=True)(raw=raw)
```
"""

from typing import TYPE_CHECKING, Any

from flyteidl2.core.artifact_id_pb2 import ArtifactKey, ArtifactVersionId

from ._card import Card, CardFormat, CardType
from ._metadata import KIND_KEY, MAX_PARENTS, Kind, Metadata
from ._partitions import Granularity, TimePartition
from ._produces import produces
from ._wrapper import ArtifactLike, new

if TYPE_CHECKING:
    from ._handle import (
        Artifact,
        ArtifactMapping,
        ArtifactRef,
        Daily,
        Hourly,
        Monthly,
        OutputPartition,
        PartitionValue,
        RequiredParam,
        TimeRange,
        Weekly,
        partition,
        required,
    )
    from ._refresh import Refresh

# Artifact handles and refresh policies are only needed by code that declares lineage. Every task pod imports this
# package (through the type engine), so they are resolved on first use instead (PEP 562).
_LAZY = {
    "Artifact": "._handle",
    "ArtifactMapping": "._handle",
    "ArtifactRef": "._handle",
    "Daily": "._handle",
    "Hourly": "._handle",
    "Monthly": "._handle",
    "OutputPartition": "._handle",
    "PartitionValue": "._handle",
    "RequiredParam": "._handle",
    "TimeRange": "._handle",
    "Weekly": "._handle",
    "partition": "._handle",
    "required": "._handle",
    "Refresh": "._refresh",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "KIND_KEY",
    "MAX_PARENTS",
    "Artifact",
    "ArtifactKey",
    "ArtifactLike",
    "ArtifactMapping",
    "ArtifactRef",
    "ArtifactVersionId",
    "Card",
    "CardFormat",
    "CardType",
    "Daily",
    "Granularity",
    "Hourly",
    "Kind",
    "Metadata",
    "Monthly",
    "OutputPartition",
    "PartitionValue",
    "Refresh",
    "RequiredParam",
    "TimePartition",
    "TimeRange",
    "Weekly",
    "new",
    "partition",
    "produces",
    "required",
]
