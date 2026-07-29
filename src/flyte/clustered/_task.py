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

    Supplies the clustered ``type``/``task_type_version`` and ``custom`` proto payload, and routes
    the container to the dedicated ``clustered`` runtime entrypoint (which sets up the torchrun
    rendezvous) instead of ``a0`` — all via generic hooks, so the serializer (``get_proto_task``)
    needs no clustered-specific branches.
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

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        # Replace the `a0` worker command with the `clustered` launcher (sibling console script).
        # The launcher derives the torchrun rendezvous from JobSet env vars and execs `torchrun ... -- a0`,
        # so each worker is the standard `a0` entrypoint (which disables the controller under torchrun).
        args = super().container_args(serialize_context)
        return ["clustered", *args[1:]] if args and args[0] == "a0" else args


TaskPluginRegistry.register(_ClusteredPlugin, ClusteredTaskTemplate)
