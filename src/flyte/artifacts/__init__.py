"""
Artifacts module

This module provides a wrapper method to mark certain outputs as artifacts with associated metadata.

Usage example:
```python
import flyte.artifacts as artifacts

@env.task
def my_task() -> MyType:
    result = MyType(...)
    metadata = artifacts.Metadata(name="my_artifact", version="1.0", description="An example artifact")
    return artifacts.new(result, metadata)
```

Launching with known artifacts:
```python
flyte.run(main, x=flyte.remote.Artifact.get("name", version="1.0"))
```

Retireve a set of artifacts and pass them as a list
```python
from flyte.remote import Artifact
flyte.run(main, x=[Artifact.get("name1", version="1.0"), Artifact.get("name2", version="2.0")])
```
OR
```python
flyte.run(main, x=flyte.remote.Artifact.list("name_prefix", partition_match="x"))
```
"""

from ._wrapper import Artifact, Metadata, new

__all__ = ["Artifact", "Metadata", "new"]
