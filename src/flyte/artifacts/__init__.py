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
"""

from ._card import Card, CardFormat, CardType
from ._metadata import KIND_KEY, ArtifactParent, Kind, Metadata
from ._wrapper import Artifact, new

__all__ = ["KIND_KEY", "Artifact", "ArtifactParent", "Card", "CardFormat", "CardType", "Kind", "Metadata", "new"]
