from __future__ import annotations

import pathlib
from typing import List, Optional

from flyteidl2.task import task_definition_pb2

from flyte._task import TaskTemplate
from flyte._task_environment import TaskEnvironment
from flyte.models import SerializationContext
from flyte.syncify import syncify

# Placeholder version stamped onto the tenant-neutral template. The consuming Go
# service overrides the real TaskIdentifier.version (a content hash) and the
# per-tenant --version code-bundle arg at registration time.
_PLACEHOLDER_VERSION = "serialized"


def _default_ctx() -> SerializationContext:
    return SerializationContext(
        version=_PLACEHOLDER_VERSION,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )


def serialize(
    task: TaskTemplate, ctx: Optional[SerializationContext] = None
) -> task_definition_pb2.TaskSpec:
    """Translate a single task to its wire TaskSpec, offline and code-agnostic.

    Mirrors what the SDK does on the dry-run deploy path, minus client and
    image-cache dependencies, so the result can be committed and bound to a
    tenant later.
    """
    from flyte._internal.runtime.convert import convert_upload_default_inputs
    from flyte._internal.runtime.task_serde import translate_task_to_wire

    resolved = ctx or _default_ctx()

    # Extract default inputs from the task interface
    default_inputs = syncify(convert_upload_default_inputs)(task.interface)

    return translate_task_to_wire(task, resolved, default_inputs=default_inputs)


def serialize_env(
    env: TaskEnvironment, ctx: Optional[SerializationContext] = None
) -> List[task_definition_pb2.TaskSpec]:
    """Serialize every task in an environment."""
    resolved = ctx or _default_ctx()
    return [serialize(t, resolved) for t in env.tasks.values()]
