"""`flyte.sandbox.orchestrator`: orchestrators that run on a sandbox leaseworker.

The decorated function only calls remote tasks. Its source travels in the task
template, so the backend runs it directly in a Monty sandbox -- no container,
no image and no code bundle -- and launches each task call as a child action.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, overload

from flyte._cache import CacheRequest
from flyte.models import NativeInterface, SerializationContext

from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate

ORCHESTRATOR_TASK_TYPE = "sandbox-orchestrator"


@dataclass(kw_only=True)
class OrchestratorTaskTemplate(SandboxedTaskTemplate):
    """A sandboxed orchestrator whose source is shipped in the task template.

    Run remotely, a sandbox leaseworker executes the source and launches the
    tasks it calls. Run locally, it behaves like any `SandboxedTaskTemplate`.
    """

    task_type: str = ORCHESTRATOR_TASK_TYPE
    task_type_version: int = 1

    child_queue: Optional[str] = None
    """Queue for the tasks this orchestrator calls. The orchestrator's own
    queue routes to a sandbox leaseworker, which cannot run them."""

    _resolved_tasks: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        from flyte.remote._task import LazyEntity

        self._check_refs_are_remote(LazyEntity)

    def _check_refs_are_remote(self, lazy_entity_type: type) -> None:
        local = [
            name
            for refs in self._external_refs.values()
            for name, ref in refs.items()
            if not isinstance(ref, lazy_entity_type)
        ]
        if local:
            import flyte.errors

            raise flyte.errors.RuntimeUserError(
                "BadConfig",
                f"flyte.sandbox.orchestrator '{self.name}' may only call remote tasks obtained with "
                f"flyte.remote.Task.get(), but {', '.join(sorted(local))} "
                f"{'is' if len(local) == 1 else 'are'} defined locally. "
                "Use env.sandbox.orchestrator for orchestrators that call local tasks.",
            )

    def runs_in_container(self) -> bool:
        return False

    @property
    def source_version(self) -> str:
        """Version derived from the source, so unchanged source keeps its version."""
        return hashlib.sha256(self._source_code.encode("utf-8")).hexdigest()[:32]

    async def resolve_tasks(self) -> None:
        """Fetch the tasks the source calls, pinning each to a concrete version."""
        for name, ref in self._external_refs["task_refs"].items():
            if name in self._resolved_tasks:
                continue
            task_id = (await ref.fetch.aio()).pb2.task_id
            self._resolved_tasks[name] = {
                "project": task_id.project,
                "domain": task_id.domain,
                "name": task_id.name,
                "version": task_id.version,
            }

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        unresolved = set(self._external_refs["task_refs"]) - set(self._resolved_tasks)
        if unresolved:
            import flyte.errors

            raise flyte.errors.RuntimeSystemError(
                "UnresolvedTasks",
                f"Orchestrator '{self.name}' was serialized before its tasks were resolved: "
                f"{', '.join(sorted(unresolved))}",
            )
        custom: Dict[str, Any] = {
            "source": self._source_code,
            "input_names": list(self._input_names),
            "tasks": dict(self._resolved_tasks),
        }
        if self.child_queue:
            custom["child_queue"] = self.child_queue
        return custom


@overload
def orchestrator(_func: Callable, /) -> OrchestratorTaskTemplate: ...


@overload
def orchestrator(
    _func: None = None,
    /,
    *,
    name: Optional[str] = None,
    queue: Optional[str] = None,
    child_queue: Optional[str] = None,
    timeout_ms: int = 30_000,
    max_stack_depth: int = 256,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Callable[[Callable], OrchestratorTaskTemplate]: ...


def orchestrator(
    _func: Optional[Callable] = None,
    /,
    *,
    name: Optional[str] = None,
    queue: Optional[str] = None,
    child_queue: Optional[str] = None,
    timeout_ms: int = 30_000,
    max_stack_depth: int = 256,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Union[OrchestratorTaskTemplate, Callable[[Callable], OrchestratorTaskTemplate]]:
    """Turn a function that only calls remote tasks into a sandboxed orchestrator.

    The function's source is shipped in the task template and executed by a
    sandbox leaseworker, so a run needs no image and starts without a pod:

    ```python
    add = flyte.remote.Task.get("math.add", auto_version="latest")

    @flyte.sandbox.orchestrator(queue="sandbox")
    def pipeline(x: int, y: int) -> int:
        return add(add(x, y), 1)

    run = flyte.run(pipeline, x=1, y=2)
    ```

    Tasks are discovered from the function's globals and must be remote
    references (`flyte.remote.Task.get`); call them without `await`.

    Args:
        name: Task name. Defaults to `<module>.<function>`.
        queue: Queue routed to a sandbox leaseworker.
        child_queue: Queue for the tasks the orchestrator calls. Defaults to
            the leaseworker's configured queue.
        timeout_ms: Time the source may spend executing, and separately the
            total it may sleep. Time spent waiting for tasks is not counted.
        max_stack_depth: Maximum recursion depth of the source.
        cache: Cache policy for the orchestrator itself.
        retries: Number of retries for the orchestrator itself.
    """

    def decorator(func: Callable) -> OrchestratorTaskTemplate:
        return OrchestratorTaskTemplate(
            func=func,
            name=name or f"{func.__module__}.{func.__name__}",
            interface=NativeInterface.from_callable(func),
            plugin_config=SandboxedConfig(timeout_ms=timeout_ms, max_stack_depth=max_stack_depth),
            image=None,
            cache=cache,
            retries=retries,
            queue=queue,
            child_queue=child_queue,
        )

    if _func is None:
        return decorator
    return decorator(_func)
