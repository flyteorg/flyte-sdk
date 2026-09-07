"""Multi-output artifact producer using the bare-tuple return annotation.

Deliberately kept in its own module *without* `from __future__ import annotations`.
Under PEP 563 every annotation becomes a string, and `typing.get_type_hints` rejects
a string that evaluates to a tuple ("Forward references must evaluate to types"), so
`-> (File, int)` only works in a module that evaluates its annotations eagerly.
`tuple[File, int]` works either way and is what the rest of the suite uses.
"""

import flyte
import flyte.artifacts as artifacts
from flyte.io import File

env = flyte.TaskEnvironment(name="bare-tuple-artifacts-test")


@env.task(produces_artifacts=True)
async def bare_tuple_task() -> (File, int):
    return artifacts.new(File(path="s3://bucket/weights.pt"), artifacts.Metadata(name="bare-model")), 7
