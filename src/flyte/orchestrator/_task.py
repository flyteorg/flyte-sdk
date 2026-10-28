"""`flyte.orchestrator.task`: tasks that orchestrate other tasks without a pod of their own.

The decorated function only calls other tasks and waits on conditions. Its
source travels in the task template, so the backend runs it directly in a
Monty sandbox -- no container for the orchestrator itself -- and launches each
task call, and each condition, as a child action.

The tasks it calls can be remote references (`flyte.remote.Task.get`), which
need nothing built, or tasks defined locally, whose image and code bundle are
built when the orchestrator is run and whose specs travel with it.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Union, overload

from flyte._cache import CacheRequest
from flyte.models import NativeInterface, SerializationContext
from flyte.sandbox._config import SandboxedConfig
from flyte.sandbox._task import SandboxedTaskTemplate

from ._conditions import LocalConditions, find_condition_refs, prelude

if TYPE_CHECKING:
    from flyteidl2.task import task_definition_pb2

    from flyte._task import TaskTemplate

    SerializeLocalTask = Callable[[TaskTemplate], Awaitable[task_definition_pb2.TaskSpec]]

ORCHESTRATOR_TASK_TYPE = "orchestrator-task"


@dataclass(kw_only=True)
class OrchestratorTaskTemplate(SandboxedTaskTemplate):
    """An orchestrator task whose source is shipped in the task template.

    Run remotely, a sandbox leaseworker executes the source and launches the
    tasks it calls. Its template is marked for that kind of worker
    (`worker_kind`), and the scheduler sends it only to such a worker, whatever
    queue it runs on. Run locally, it behaves like any
    `SandboxedTaskTemplate`.
    """

    task_type: str = ORCHESTRATOR_TASK_TYPE
    task_type_version: int = 1
    runs_in_container: bool = field(default=False, init=False, repr=False, compare=False)
    worker_kind: str = field(default="sandbox", init=False, repr=False, compare=False)

    _resolved_tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    # Whether the source uses `flyte.new_condition`, and so ships with the condition prelude.
    _uses_conditions: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # In an `async def` orchestrator a task call is awaited, as in any async task, so
        # `asyncio.gather` runs several at once.
        self._awaitable_tasks = inspect.iscoroutinefunction(self.func)
        bindings = find_condition_refs(self.func, self._external_refs["trace_refs"])
        if bindings:
            self._uses_conditions = True
            self._has_external_refs = True
            self._source_code = prelude(bindings) + self._source_code
        self._check_refs_are_tasks()

    def _check_refs_are_tasks(self) -> None:
        other = sorted(name for kind in ("trace_refs", "durable_refs") for name in self._external_refs[kind])
        if other:
            import flyte.errors

            raise flyte.errors.RuntimeUserError(
                "BadConfig",
                f"flyte.orchestrator.task '{self.name}' may only call tasks and flyte.new_condition, but "
                f"{', '.join(other)} {'is' if len(other) == 1 else 'are'} neither. "
                "Use env.orchestrator.task for orchestrators that call traces or durable operations.",
            )

    async def _run_sandboxed(self, *args, **kwargs) -> Any:
        """Run locally, answering the condition prelude's calls with `flyte.new_condition`."""
        if not self._uses_conditions:
            return await super()._run_sandboxed(*args, **kwargs)

        from flyte.sandbox._bridge import ExternalFunctionBridge

        inputs = self._build_inputs(*args, **kwargs)
        trace_refs = {**self._external_refs["trace_refs"], **LocalConditions().functions()}
        bridge = ExternalFunctionBridge(
            task_refs=self._external_refs["task_refs"],
            trace_refs=trace_refs,
            durable_refs=self._external_refs["durable_refs"],
            awaitable_tasks=self._awaitable_tasks,
        )
        return await bridge.execute_monty(self.plugin_config, self._source_code, inputs)

    @property
    def source_version(self) -> str:
        """Version derived from the source and the versions of the tasks it calls, so an
        orchestrator that would behave the same keeps its version."""
        digest = hashlib.sha256(self._source_code.encode("utf-8"))
        for name in sorted(self._resolved_tasks):
            digest.update(f"{name}={self._resolved_tasks[name]['version']}".encode("utf-8"))
        return digest.hexdigest()[:32]

    async def resolve_tasks(self, serialize_local: Optional[SerializeLocalTask] = None) -> None:
        """Pin every task the source calls to what the worker should run.

        A remote task (`flyte.remote.Task.get`) is pinned to the version it resolves to; the
        worker fetches its spec from the backend. A local task has no deployed spec, so
        *serialize_local* builds one (which means building its image and code bundle) and the
        spec travels in the template, with its input names in declared order so the worker
        can bind positional arguments. A remote task's order is not known, so it is called
        with keyword arguments unless it has a single input.
        """
        from flyte.remote._task import LazyEntity

        for name, ref in self._external_refs["task_refs"].items():
            if isinstance(ref, LazyEntity):
                if name in self._resolved_tasks:
                    continue
                task_id = (await ref.fetch.aio()).pb2.task_id
                self._resolved_tasks[name] = {
                    "project": task_id.project,
                    "domain": task_id.domain,
                    "name": task_id.name,
                    "version": task_id.version,
                }
                continue

            if serialize_local is None:
                import flyte.errors

                raise flyte.errors.RuntimeSystemError(
                    "UnresolvedTasks",
                    f"Orchestrator '{self.name}' calls the local task '{name}', which has to be "
                    "serialized when the orchestrator is run.",
                )
            # Serialized on every run: the spec follows the task's current code and image.
            spec = await serialize_local(ref)
            task_id = spec.task_template.id
            self._resolved_tasks[name] = {
                "project": task_id.project,
                "domain": task_id.domain,
                "name": task_id.name,
                "version": task_id.version,
                "spec": base64.b64encode(spec.SerializeToString(deterministic=True)).decode("ascii"),
                # The spec sorts its inputs by name; positional arguments bind in this order.
                "input_names": list(ref.interface.inputs),
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
        config = self.plugin_config or SandboxedConfig()
        return {
            "source": self._source_code,
            "input_names": list(self._input_names),
            "tasks": dict(self._resolved_tasks),
            # A task call returns a future for the source to await, as in any async task.
            "awaitable_tasks": self._awaitable_tasks,
            # The worker applies these, capped at its own limits. Memory is not
            # the user's to set: the worker gives every orchestrator the same.
            "timeout_ms": config.timeout_ms,
            "max_stack_depth": config.max_stack_depth,
        }


@overload
def task(_func: Callable, /) -> OrchestratorTaskTemplate: ...


@overload
def task(
    _func: None = None,
    /,
    *,
    name: Optional[str] = None,
    queue: Optional[str] = None,
    timeout_ms: int = 30_000,
    max_stack_depth: int = 256,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Callable[[Callable], OrchestratorTaskTemplate]: ...


def task(
    _func: Optional[Callable] = None,
    /,
    *,
    name: Optional[str] = None,
    queue: Optional[str] = None,
    timeout_ms: int = 30_000,
    max_stack_depth: int = 256,
    cache: CacheRequest = "disable",
    retries: int = 0,
) -> Union[OrchestratorTaskTemplate, Callable[[Callable], OrchestratorTaskTemplate]]:
    """Turn a function that only calls other tasks into an orchestrator task.

    The function's source is shipped in the task template and executed by a
    sandbox leaseworker, so the orchestrator itself starts without a pod:

    ```python
    add = flyte.remote.Task.get("math.add", auto_version="latest")

    @flyte.orchestrator.task
    def pipeline(x: int, y: int) -> int:
        return add(add(x, y), 1)

    run = flyte.run(pipeline, x=1, y=2)
    ```

    Tasks are discovered from the function's globals; call them without
    `await`. They can be:

    - remote references (`flyte.remote.Task.get`). Nothing is built, so the run
      is submitted straight away.
    - tasks defined locally with `@env.task`. Their image and code bundle are
      built when the orchestrator is run, as for any `flyte.run`, and their
      specs travel in the orchestrator's template.

    It can also wait for a value someone supplies from the UI or the API, with
    `flyte.new_condition` and the condition's `wait()`, again without `await`.
    While it waits, nothing runs and no pod is held:

    ```python
    @flyte.orchestrator.task
    def deploy(model: str) -> str:
        report = evaluate(model)
        approval = flyte.new_condition("approve", prompt=f"Deploy {model}? {report}", timeout=86_400)
        if approval.wait():
            return publish(model)
        return "rejected"
    ```

    `data_type` is one of bool, int, float and str, and `timeout` is in seconds
    or a `timedelta`, imported inside the function since only its source is
    shipped. Webhooks are not supported here.

    A condition that is not signalled in time raises
    `flyte.errors.ConditionTimedoutError` from `wait()`, and one that fails or
    is aborted raises `flyte.errors.ConditionFailedError` or
    `flyte.errors.ActionAbortedError`. The orchestrator runs without the SDK's
    exception classes, so inside it these are the builtins `TimeoutError`,
    `RuntimeError` and `RuntimeError`: `except
    flyte.errors.ConditionTimedoutError` catches only timeouts, but `except
    RuntimeError` does not catch them.

    Args:
        name: Task name. Defaults to `<module>.<function>`.
        queue: Queue to run on, as for any task. Defaults to the run's queue.
            The scheduler picks a sandbox leaseworker on the queue's clusters
            by the task type, and the tasks the orchestrator calls run on the
            same queue.
        timeout_ms: Time the source may spend executing, and separately the
            total it may sleep. Time spent waiting for tasks is not counted.
        max_stack_depth: Maximum recursion depth of the source.
        cache: Cache policy for the orchestrator itself.
        retries: Number of retries for the orchestrator itself.
    """

    def decorator(func: Callable) -> OrchestratorTaskTemplate:
        return OrchestratorTaskTemplate(
            func=func,
            name=name or f"{func.__module__}.{getattr(func, '__name__')}",
            interface=NativeInterface.from_callable(func),
            plugin_config=SandboxedConfig(timeout_ms=timeout_ms, max_stack_depth=max_stack_depth),
            image=None,
            cache=cache,
            retries=retries,
            queue=queue,
        )

    if _func is None:
        return decorator
    return decorator(_func)
