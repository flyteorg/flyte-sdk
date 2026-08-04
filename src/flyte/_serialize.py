from __future__ import annotations

import pathlib
from typing import List, Optional

from flyteidl2.task import task_definition_pb2

from flyte._task import TaskTemplate
from flyte._task_environment import TaskEnvironment
from flyte.models import SerializationContext
from flyte.syncify import syncify

# Default version stamped onto the TaskSpec when the caller does not supply one
# through a SerializationContext.
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

    Builds the TaskSpec without a client, image cache, or code bundle, so it can
    be produced ahead of time. Pass a SerializationContext to override the
    defaults.
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
