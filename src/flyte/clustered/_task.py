from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, cast

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext

if TYPE_CHECKING:
    from flyte.clustered._environment import ClusteredTaskEnvironment


@dataclass(frozen=True)
class _ClusteredPlugin:
    """Marker config that selects ``ClusteredTaskTemplate`` via the task plugin registry.

    Mirrors ``flyte.extras._sleep.Sleep`` — it carries no data; the clustered settings live on
    the ``ClusteredTaskEnvironment`` and are read back through ``parent_env`` at serialize time.
    """


@dataclass(kw_only=True)
class ClusteredTaskTemplate(AsyncFunctionTaskTemplate):
    """Task template for ``ClusteredTaskEnvironment``.

    Supplies the clustered ``type``/``task_type_version``, the ``custom`` proto payload, and the
    entrypoint-wrapper ``command`` so that the generic serializer (``get_proto_task``) needs no
    clustered-specific branches.
    """

    plugin_config: _ClusteredPlugin
    task_type: str = "clustered-task"
    task_type_version: int = 1

    def custom_config(self, sctx: SerializationContext) -> Dict:
        # parent_env is a weakref set by TaskEnvironment.task(); it is alive during serialization.
        env = self.parent_env() if self.parent_env else None
        if env is None:
            return {}
        return cast("ClusteredTaskEnvironment", env).to_custom_dict()

    def container_command(self, sctx: SerializationContext) -> List[str]:
        # PID-1 is the entrypoint wrapper; the original a0 invocation lives in container_args and
        # is passed through to torchrun -> a0.
        return ["python", "-m", "flyte.clustered._entrypoint"]


TaskPluginRegistry.register(_ClusteredPlugin, ClusteredTaskTemplate)
