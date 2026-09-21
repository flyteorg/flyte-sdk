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
The set of keys is fixed by the first partitioned version (or `Artifact.declare`). A later
version with different keys is stored as published and answers by the keys it carries;
compare it with `Artifact.get_schema(name)` to see the difference.

Producing artifacts from a task that does not wrap its outputs: the caller declares them.
```python
with artifacts.produces(o0=artifacts.Metadata(name="events", partitions={"date": day})):
    await clean.override(produces_artifacts=True)(raw=raw)
```
"""

from flyteidl2.core.artifact_id_pb2 import ArtifactKey, ArtifactVersionId

from ._card import Card, CardFormat, CardType
from ._metadata import KIND_KEY, MAX_PARENTS, Kind, Metadata
from ._partitions import Granularity, TimePartition
from ._produces import produces
from ._wrapper import Artifact, new

__all__ = [
    "KIND_KEY",
    "MAX_PARENTS",
    "Artifact",
    "ArtifactKey",
    "ArtifactVersionId",
    "Card",
    "CardFormat",
    "CardType",
    "Granularity",
    "Kind",
    "Metadata",
    "TimePartition",
    "new",
    "produces",
]
