from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from flyte.extend import AsyncFunctionTaskTemplate, TaskPluginRegistry
from flyte.models import SerializationContext

if TYPE_CHECKING:
    from flyte.clustered._environment import ClusteredSettings


@dataclass(frozen=True)
class _ClusteredPlugin:
    """Plugin config that selects `ClusteredTaskTemplate` via the task plugin registry.

    Carries the `ClusteredSettings` snapshot the task serializes from, so the template never has to
    reach back to its `ClusteredTaskEnvironment` through the `parent_env` weakref (which pickling drops).

    `settings` is None only for the in-container routing marker built by
    `_run_python_script._build_script_runner_task`, which is never serialized.
    """

    settings: Optional[ClusteredSettings] = None


@dataclass(kw_only=True)
class ClusteredTaskTemplate(AsyncFunctionTaskTemplate):
    """Task template for `ClusteredTaskEnvironment`.

    Supplies the clustered `type`/`task_type_version` and `custom` proto payload, and routes
    the container to the dedicated `clustered` runtime entrypoint (which sets up the torchrun
    rendezvous) instead of `a0` — all via generic hooks, so the serializer (`get_proto_task`)
    needs no clustered-specific branches.
    """

    plugin_config: _ClusteredPlugin
    task_type: str = "clustered-task"
    task_type_version: int = 1

    def _settings(self) -> ClusteredSettings:
        settings = self.plugin_config.settings
        if settings is None:
            # Never serialize a settings-less marker: an empty `custom` registers fine and only fails
            # in the backend at execution time (`invalid ClusteredTaskSpec: nil Struct Object passed`).
            import flyte.errors

            raise flyte.errors.RuntimeUserError(
                "BadConfig",
                f"Task {self.name} has no clustered settings to serialize: build it from a "
                "ClusteredTaskEnvironment rather than with a bare _ClusteredPlugin().",
            )
        return settings

    def custom_config(self, sctx: SerializationContext) -> Dict:
        return self._settings().to_custom_dict()

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

        launcher_args: List[str] = []
        runtime = self._settings().runtime
        if not isinstance(runtime, TorchRun):
            launcher_args.append(f"--runtime={launcher_name(runtime)}")
        return ["clustered", *launcher_args, *args[1:]]


TaskPluginRegistry.register(_ClusteredPlugin, ClusteredTaskTemplate)
