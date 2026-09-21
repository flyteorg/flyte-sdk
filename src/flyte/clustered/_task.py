from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, cast

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext

if TYPE_CHECKING:
    from flyte.clustered._environment import MultiNodeTaskEnvironment


@dataclass(frozen=True)
class _ClusteredPlugin:
    """Marker config that selects `ClusteredTaskTemplate` via the task plugin registry.

    Mirrors `flyte.extras._sleep.Sleep` — it carries no data; the clustered settings live on
    the `MultiNodeTaskEnvironment` and are read back through `parent_env` at serialize time.
    """


@dataclass(kw_only=True)
class ClusteredTaskTemplate(AsyncFunctionTaskTemplate):
    """Task template for `MultiNodeTaskEnvironment`.

    Supplies the clustered `type`/`task_type_version` and `custom` proto payload, and routes
    the container to the dedicated `clustered` runtime entrypoint (which sets up the torchrun
    rendezvous) instead of `a0` — all via generic hooks, so the serializer (`get_proto_task`)
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
        return cast("MultiNodeTaskEnvironment", env).to_custom_dict()

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        # Replace the `a0` worker command with the `clustered` launcher (sibling console script).
        # The launcher derives the process topology from JobSet env vars and execs the runtime selected
        # by `--runtime` with `a0` as the worker command, so each worker is the standard `a0` entrypoint
        # (which runs with no controller under a clustered launcher). `--runtime` is emitted only for
        # non-default runtimes so images carrying an older launcher keep working for torchrun, and an
        # older launcher rejects it loudly instead of silently running torchrun.
        args = super().container_args(serialize_context)
        if not args or args[0] != "a0":
            return args
        from flyte.clustered._environment import TorchRun, launcher_name

        env = self.parent_env() if self.parent_env else None
        launcher_args: List[str] = []
        if env is not None:
            runtime = cast("MultiNodeTaskEnvironment", env).runtime
            if not isinstance(runtime, TorchRun):
                launcher_args.append(f"--runtime={launcher_name(runtime)}")
        return ["clustered", *launcher_args, *args[1:]]


TaskPluginRegistry.register(_ClusteredPlugin, ClusteredTaskTemplate)
