from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import rich.repr

from ._cache import Cache, CacheRequest
from ._doc import Documentation
from ._environment import Environment
from ._image import Image
from ._link import Link
from ._logging import logger
from ._pod import PodTemplate
from ._resources import Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._secret import SecretRequest
from ._task import AsyncFunctionTaskTemplate, TaskTemplate
from ._trigger import Trigger
from .models import MAX_INLINE_IO_BYTES, NativeInterface

if TYPE_CHECKING:
    from ._task import F, P, R
    from .sandbox._code_task import CodeTaskTemplate
    from .sandbox._task import SandboxedTaskTemplate


@rich.repr.auto
@dataclass(init=True, repr=True)
class TaskEnvironment(Environment):
    """
    Define an execution environment for a set of tasks.

    Task configuration in Flyte has three levels (most general to most specific):

    1. **TaskEnvironment** — sets defaults for all tasks in the environment
    2. **@env.task decorator** — overrides per-task settings
    3. **task.override()** — overrides at invocation time

    For shared parameters, the more specific level overrides the more general one.

    Example:

    ```python
    env = flyte.TaskEnvironment(
        name="my_env",
        image=flyte.Image.from_debian_base(python="3.12").with_pip_packages("pandas"),
        resources=flyte.Resources(cpu="1", memory="1Gi"),
    )

    @env.task
    async def my_task():
        pass
    ```

    **Parameter interaction across configuration levels:**

    | Parameter | `TaskEnvironment` | `@env.task` | `task.override()` |
    |-----------|:-----------------:|:-----------:|:-----------------:|
    | `name` | Yes (required) | — | — |
    | `image` | Yes | — | — |
    | `depends_on` | Yes | — | — |
    | `description` | Yes | — | — |
    | `plugin_config` | Yes | — | — |
    | `resources` | Yes | — | Yes* |
    | `env_vars` | Yes | — | Yes* |
    | `secrets` | Yes | — | Yes* |
    | `cache` | Yes | Yes | Yes |
    | `pod_template` | Yes | Yes | Yes |
    | `reusable` | Yes | — | Yes |
    | `interruptible` | Yes | Yes | Yes |
    | `queue` | Yes | Yes | Yes |
    | `short_name` | — | Yes | Yes |
    | `retries` | — | Yes | Yes |
    | `timeout` | — | Yes | Yes |
    | `max_inline_io_bytes` | — | Yes | Yes |
    | `links` | — | Yes | Yes |
    | `report` | — | Yes | — |
    | `triggers` | — | Yes | — |
    | `docs` | — | Yes | — |

    *When `reusable` is set, `resources`, `env_vars`, and `secrets` can only
    be overridden via `task.override()` with `reusable="off"` in the same call.

    :param name: Name of the environment (required). Must be snake_case or kebab-case.
        TaskEnvironment level only. The fully-qualified name of each task is
        `<env_name>.<function_name>` (e.g., environment `"my_env"` containing
        function `my_task` produces FQN `"my_env.my_task"`). Neither component
        is overridable.
    :param image: Docker image for the environment. Can be a string (image URI),
        an `Image` object, or `"auto"` to use the default image.
        TaskEnvironment level only.
    :param depends_on: List of other environments this one depends on. Used at deploy time
        to ensure dependencies are also deployed. TaskEnvironment level only.
    :param description: Human-readable description (max 255 characters).
        TaskEnvironment level only.
    :param plugin_config: Plugin configuration for custom task types (e.g., Ray, Spark).
        Cannot be combined with `reusable`. TaskEnvironment level only.
    :param resources: Compute resources (CPU, memory, GPU, disk). Overridable via
        `task.override(resources=...)` when not using reusable containers.
    :param env_vars: Environment variables as `dict[str, str]`. Overridable via
        `task.override(env_vars=...)` when not using reusable containers.
    :param secrets: Secrets to inject. Overridable via `task.override(secrets=...)`
        when not using reusable containers.
    :param cache: Cache policy — `"auto"`, `"override"`, `"disable"`, or a `Cache` object.
        Also settable in `@env.task(cache=...)` and `task.override(cache=...)`.
    :param reusable: `ReusePolicy` for container reuse. Also overridable via
        `task.override(reusable=...)`. Note: when `reusable` is set on the
        environment, overriding `resources`, `env_vars`, or `secrets` in
        `task.override()` requires passing `reusable="off"` in the same call.
        Additionally, `secrets` cannot be overridden at the `@env.task`
        decorator level when the environment has `reusable` set.
    :param queue: Queue name for scheduling. Queues identify specific partitions
        of your compute infrastructure (e.g., a particular cluster in a
        multi-cluster deployment) and are configured as part of your Flyte/Union
        deployment. Also settable in `@env.task` and `task.override`.
    :param pod_template: Kubernetes pod template for advanced configuration (sidecars,
        volumes, etc.). Also settable in `@env.task` and `task.override`.
    :param interruptible: Whether tasks can run on spot/preemptible instances. Also
        settable in `@env.task` and `task.override`.
    """

    cache: CacheRequest = "disable"
    reusable: ReusePolicy | None = None
    plugin_config: Optional[Any] = None
    queue: Optional[str] = None

    _tasks: Dict[str, TaskTemplate] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reusable is not None and self.plugin_config is not None:
            raise ValueError("Cannot set plugin_config when environment is reusable.")
        if self.reusable and not isinstance(self.reusable, ReusePolicy):
            raise TypeError(f"Expected reusable to be of type ReusePolicy, got {type(self.reusable)}")
        if self.cache and not isinstance(self.cache, (str, Cache)):
            raise TypeError(f"Expected cache to be of type str or Cache, got {type(self.cache)}")

        # Synthesize a hidden __prewarm__ task so env.prewarm() has something cheap to
        # submit. The task must share the env's image / ReusePolicy / env_vars / secrets
        # so its version hash matches the env's real tasks and lands on the same pool.
        if self.reusable is not None:
            self._register_prewarm_task()

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env_vars: Optional[Dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[List[Environment]] = None,
        description: Optional[str] = None,
        interruptible: Optional[bool] = None,
        include: Optional[Tuple[str, ...]] = None,
        **kwargs: Any,
    ) -> TaskEnvironment:
        """
        Create a new `TaskEnvironment` that shares most settings with this one
        but differs in name and selected overrides.

        Use `clone_with` when you need several environments that share a common
        base configuration (image, resources, secrets, etc.) but vary in one or
        two settings, avoiding repetition.

        ```python
        gpu_env = flyte.TaskEnvironment(
            name="gpu_env",
            image=my_image,
            resources=flyte.Resources(gpu="A100:1", memory="16Gi"),
        )

        # Same image and resources, different name and cache policy
        gpu_env_cached = gpu_env.clone_with("gpu_env_cached", cache="auto")
        ```

        Any parameter not explicitly passed inherits the value from the
        original environment.

        :param name: Name for the new environment (required).
        :param image: Override the container image.
        :param resources: Override compute resources.
        :param env_vars: Override environment variables.
        :param secrets: Override secrets.
        :param depends_on: Override deployment dependencies.
        :param description: Override the description.
        :param interruptible: Override the interruptible setting.
        :param kwargs: Additional `TaskEnvironment`-specific overrides
            (e.g., `cache`, `reusable`, `plugin_config`).
        """
        cache = kwargs.pop("cache", None)
        reusable = None
        reusable_set = False
        if "reusable" in kwargs:
            reusable_set = True
            reusable = kwargs.pop("reusable", None)

        # validate unknown kwargs if needed
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        if image is not None:
            kwargs["image"] = image
        if resources is not None:
            kwargs["resources"] = resources
        if cache is not None:
            kwargs["cache"] = cache
        if env_vars is not None:
            kwargs["env_vars"] = env_vars
        if reusable_set:
            kwargs["reusable"] = reusable
        if secrets is not None:
            kwargs["secrets"] = secrets
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        if description is not None:
            kwargs["description"] = description
        if interruptible is not None:
            kwargs["interruptible"] = interruptible
        if include is not None:
            kwargs["include"] = tuple(include) if not isinstance(include, tuple) else include
        return replace(self, **kwargs)

    @overload
    def task(
        self,
        *,
        short_name: Optional[str] = None,
        cache: CacheRequest | None = None,
        retries: Union[int, RetryStrategy] = 0,
        timeout: Union[timedelta, int] = 0,
        docs: Optional[Documentation] = None,
        pod_template: Optional[Union[str, PodTemplate]] = None,
        report: bool = False,
        interruptible: bool | None = None,
        max_inline_io_bytes: int = MAX_INLINE_IO_BYTES,
        queue: Optional[str] = None,
        triggers: Tuple[Trigger, ...] | Trigger = (),
        links: Tuple[Link, ...] | Link = (),
        task_resolver: Any | None = None,
        entrypoint: bool = False,
    ) -> Callable[[Callable[P, R]], AsyncFunctionTaskTemplate[P, R, Callable[P, R]]]: ...

    @overload
    def task(
        self,
        _func: Callable[P, R],
        /,
    ) -> AsyncFunctionTaskTemplate[P, R, Callable[P, R]]: ...

    def task(
        self,
        _func: F | None = None,
        *,
        short_name: Optional[str] = None,
        cache: CacheRequest | None = None,
        retries: Union[int, RetryStrategy] = 0,
        timeout: Union[timedelta, int] = 0,
        docs: Optional[Documentation] = None,
        pod_template: Optional[Union[str, PodTemplate]] = None,
        report: bool = False,
        interruptible: bool | None = None,
        max_inline_io_bytes: int = MAX_INLINE_IO_BYTES,
        queue: Optional[str] = None,
        triggers: Tuple[Trigger, ...] | Trigger = (),
        links: Tuple[Link, ...] | Link = (),
        task_resolver: Any | None = None,
        entrypoint: bool = False,
    ) -> Callable[[F], AsyncFunctionTaskTemplate[P, R, F]] | AsyncFunctionTaskTemplate[P, R, F]:
        """
        Decorate a function to be a task.

        :param _func: Optional The function to decorate. If not provided, the decorator will return a callable that
        accepts a function to be decorated.
        :param short_name: Optional friendly name for the task or action, used in
            parts of the UI (defaults to the function name). Overriding `short_name`
            does not change the task's fully-qualified name.
        :param cache: Optional The cache policy for the task, defaults to auto, which will cache the results of the
        task.
        :param retries: Number of retries (`int`) or a `RetryStrategy` object that
            defines retry behavior. Defaults to `0` (no retries).
        :param docs: Optional The documentation for the task, if not provided the function docstring will be used.
        :param timeout: Task timeout, as a `timedelta` object or an integer number
            of seconds. `0` means no timeout.
        :param pod_template: Optional The pod template for the task, if not provided the default pod template will be
        used.
        :param report: Optional Whether to generate the html report for the task, defaults to False.
        :param max_inline_io_bytes: Maximum allowed size (in bytes) for all inputs and
            outputs passed directly to the task (e.g., primitives, strings, dicts).
            Does not apply to files, directories, or dataframes. Default is 10 MiB.
        :param triggers: Optional A tuple of triggers to associate with the task. This allows the task to be run on a
         schedule or in response to events. Triggers can be defined using the `flyte.trigger` module.
        :param links: Optional A tuple of links to associate with the task. Links can be used to provide
         additional context or information about the task. Links should implement the `flyte.Link` protocol
        :param interruptible: Optional Whether the task is interruptible, defaults to environment setting.
        :param queue: Optional queue name to use for this task. If not set, the environment's queue will be used.
        :param entrypoint: Optionally mark a task as an entrypoint task, defaults to False. This serves as a hint to
            the UI.
        :param task_resolver: Optional TaskResolver protocol to load tasks using custom policy.

        :return: A TaskTemplate that can be used to deploy the task.
        """
        from ._task import F, P, R

        if self.reusable is not None:
            if pod_template is not None:
                raise ValueError("Cannot set pod_template when environment is reusable.")

        def decorator(func: F) -> AsyncFunctionTaskTemplate[P, R, F]:
            short = short_name or func.__name__
            task_name = self.name + "." + func.__name__

            if not inspect.iscoroutinefunction(func) and self.reusable is not None:
                if self.reusable.concurrency > 1:
                    raise ValueError(
                        "Reusable environments with concurrency greater than 1 are only supported for async tasks. "
                        "Please use an async function or set concurrency to 1."
                    )

            if self.plugin_config is not None:
                from flyte.extend import TaskPluginRegistry

                task_template_class: type[AsyncFunctionTaskTemplate[P, R, F]] | None = TaskPluginRegistry.find(
                    config_type=type(self.plugin_config)
                )
                if task_template_class is None:
                    raise ValueError(
                        f"No task plugin found for config type {type(self.plugin_config)}. "
                        f"Please register a plugin using flyte.extend.TaskPluginRegistry.register() api."
                    )
            else:
                task_template_class = AsyncFunctionTaskTemplate[P, R, F]

            task_template_class = cast(type[AsyncFunctionTaskTemplate[P, R, F]], task_template_class)
            tmpl = task_template_class(
                func=func,
                name=task_name,
                image=self.image,
                resources=self.resources,
                cache=cache or self.cache,
                retries=retries,
                timeout=timeout,
                reusable=self.reusable,
                docs=docs,
                env_vars=self.env_vars,
                secrets=self.secrets,
                pod_template=pod_template or self.pod_template,
                parent_env=weakref.ref(self),
                parent_env_name=self.name,
                interface=NativeInterface.from_callable(func),
                report=report,
                short_name=short,
                plugin_config=self.plugin_config,
                max_inline_io_bytes=max_inline_io_bytes,
                queue=queue or self.queue,
                interruptible=interruptible if interruptible is not None else self.interruptible,
                entrypoint=entrypoint,
                triggers=(triggers,) if isinstance(triggers, Trigger) else tuple(triggers),
                links=tuple(links) if isinstance(links, (list, tuple)) else (links,),
                task_resolver=task_resolver,
            )
            self._tasks[task_name] = tmpl
            return tmpl

        if _func is None:
            return cast(Callable[[F], AsyncFunctionTaskTemplate[P, R, F]], decorator)
        return cast(AsyncFunctionTaskTemplate[P, R, F], decorator(_func))

    @property
    def sandbox(self) -> _SandboxNamespace:
        """Access the sandbox namespace for creating sandboxed tasks."""
        return _SandboxNamespace(self)

    @property
    def tasks(self) -> Dict[str, TaskTemplate]:
        """
        Get all tasks defined in the environment.
        """
        return self._tasks

    def _register_prewarm_task(self) -> None:
        """Attach a hidden no-op task to this env to back env.prewarm().

        Uses a SDK-provided resolver instead of the user-code resolver so the
        worker can load the task without importing user modules.
        """
        from ._internal.resolvers.prewarm import (
            PrewarmTaskResolver,
            _prewarm_noop,
            prewarm_task_full_name,
            prewarm_task_short_name,
        )

        task_name = prewarm_task_full_name(self.name)
        tmpl = AsyncFunctionTaskTemplate[Any, int, Callable[[], Any]](
            func=_prewarm_noop,
            name=task_name,
            image=self.image,
            resources=self.resources,
            cache="disable",
            reusable=self.reusable,
            env_vars=self.env_vars,
            secrets=self.secrets,
            pod_template=self.pod_template,
            parent_env=weakref.ref(self),
            parent_env_name=self.name,
            interface=NativeInterface.from_callable(_prewarm_noop),
            short_name=prewarm_task_short_name(self.name),
            interruptible=self.interruptible,
            queue=self.queue,
            task_resolver=PrewarmTaskResolver(),
        )
        self._tasks[task_name] = tmpl

    def _resolve_prewarm_target(self) -> Optional[TaskTemplate]:
        """Run the validation gauntlet for prewarm{,_sync}().

        Returns the prewarm `TaskTemplate` if we should fire it, otherwise
        `None` (after logging the reason). Centralizes the warn-and-return
        cases shared by the async and sync entry points.
        """
        if self.reusable is None:
            logger.warning(
                f"prewarm() called on TaskEnvironment '{self.name}' which is not reusable — no-op."
            )
            return None

        from ._context import internal_ctx

        ctx = internal_ctx()
        if not ctx.is_task_context():
            logger.warning(
                f"prewarm() called on '{self.name}' outside a task context — no-op. "
                "Call this from inside a @env.task function."
            )
            return None

        try:
            from ._internal.controllers import get_controller
            from ._internal.controllers._local_controller import LocalController
        except Exception:
            logger.warning(f"prewarm() on '{self.name}': controller machinery unavailable — no-op.")
            return None

        try:
            controller = get_controller()
        except Exception:
            logger.warning(f"prewarm() on '{self.name}': no controller initialized — no-op.")
            return None

        if isinstance(controller, LocalController):
            # In local execution there is no remote replica pool to warm.
            return None

        from ._internal.resolvers.prewarm import prewarm_task_full_name

        task_name = prewarm_task_full_name(self.name)
        prewarm_task = self._tasks.get(task_name)
        if prewarm_task is None:
            # Defensive: should not happen since __post_init__ registers it for reusable envs.
            logger.warning(f"prewarm() on '{self.name}': hidden task missing — no-op.")
            return None

        # Surface the idle_ttl window so the user sees how long the pool stays
        # warm after this prewarm completes. If their setup work exceeds this,
        # the pool will scale down before the heavy task arrives.
        idle_ttl_seconds = int(self.reusable.idle_ttl.total_seconds())
        logger.info(
            f"prewarm: warming env '{self.name}' (idle_ttl={idle_ttl_seconds}s — "
            f"pool will stay alive ~{idle_ttl_seconds}s after each task completes)"
        )
        return prewarm_task

    async def prewarm(self) -> None:
        """Pre-warm this reusable env's worker pool by submitting a hidden no-op.

        The backend's first task submission to an `actor` pool triggers
        `GetOrCreateEnvironment`, which spawns up to `min_replica_count`
        workers. Subsequent tasks on the same env land on the warm pool
        with no cold-start cost.

        **Awaiting waits for completion.** `await env.prewarm()` blocks until
        the no-op sub-action terminates — i.e., the pool has reached HEALTHY
        and the no-op has actually executed on a worker. On return, the pool
        is guaranteed to be ready.

        **For fire-and-forget**, use the standard Python pattern:

        ```python
        asyncio.create_task(env.prewarm())   # schedule, do not wait
        await asyncio.sleep(setup_seconds)   # other work runs in parallel
        await heavy_task(...)                # pool is warm by now
        ```

        In sync tasks, use :meth:`prewarm_sync` instead.

        **Warm window.** A single prewarm only stays useful for roughly
        `ReusePolicy.idle_ttl - pod_startup_time`. If your driver does
        long work before invoking the heavy task, increase `idle_ttl`.
        Note that the same `idle_ttl` also applies *after* the heavy task
        completes, so a longer value also delays scale-down.

        **No-op cases:**

        - Called on a non-reusable environment → logs a warning and returns.
        - Running in local execution mode → silent no-op.
        - Called outside a task / run context (no controller) → warns and returns.

        Example (fire-and-forget, the common case):

        ```python
        heavy_env = flyte.TaskEnvironment(
            name="heavy",
            reusable=flyte.ReusePolicy(replicas=(2, 4), idle_ttl=600),
        )

        @heavy_env.task
        async def big_inference(x: str) -> str: ...

        driver_env = heavy_env.clone_with("driver", reusable=None, depends_on=[heavy_env])

        @driver_env.task
        async def main():
            asyncio.create_task(heavy_env.prewarm())   # kick off pool startup
            await asyncio.sleep(60)                    # other async setup work
            return await big_inference("...")          # pool already HEALTHY
        ```
        """
        prewarm_task = self._resolve_prewarm_target()
        if prewarm_task is None:
            return
        await prewarm_task.aio()  # type: ignore[misc]

    def prewarm_sync(self) -> None:
        """Synchronous companion to :meth:`prewarm` for use in sync `@env.task` functions.

        Blocks until the no-op sub-action terminates. On return the pool is
        guaranteed HEALTHY. Matches the SDK convention used by
        `File.download_sync`, `Checkpoint.load_sync`, etc.

        For fire-and-forget in sync code, wrap with a thread:

        ```python
        import threading
        threading.Thread(target=heavy_env.prewarm_sync, daemon=True).start()
        do_setup()                # warms in background
        return heavy_task(...)    # pool may or may not be ready
        ```

        Same no-op cases as :meth:`prewarm` (non-reusable, local mode,
        outside task context).
        """
        prewarm_task = self._resolve_prewarm_target()
        if prewarm_task is None:
            return
        from ._internal.controllers import get_controller

        controller = get_controller()
        fut = controller.submit_sync(prewarm_task)
        fut.result()  # block until terminal

    @classmethod
    def from_task(
        cls,
        name: str,
        *tasks: TaskTemplate,
        depends_on: Optional[List["Environment"]] = None,
    ) -> TaskEnvironment:
        """
        Create a TaskEnvironment from a list of tasks. All tasks should have the same image or no Image defined.
        Similarity of Image is determined by the python reference, not by value.

        If images are different, an error is raised. If no image is defined, the image is set to "auto".

        For any other tasks that need to be use these tasks, the returned environment can be used in the `depends_on`
        attribute of the other TaskEnvironment.

        :param name: The name of the environment.
        :param tasks: The list of tasks to create the environment from.
        :param depends_on: Optional list of environments that this environment depends on.

        :raises ValueError: If tasks are assigned to multiple environments or have different images.
        :return: The created TaskEnvironment.
        """
        envs = [t.parent_env() for t in tasks if t.parent_env and t.parent_env() is not None]
        if envs:
            raise ValueError("Tasks cannot assigned to multiple environments.")
        images = {t.image for t in tasks}
        if len(images) > 1:
            raise ValueError("Tasks must have the same image to be in the same environment.")
        image: Union[str, Image, None] = images.pop() if images else "auto"
        env = cls(name, image=image, depends_on=depends_on or [])
        for t in tasks:
            env._tasks[t.name] = t
            t.parent_env = weakref.ref(env)
            t.parent_env_name = name
        return env


class _SandboxNamespace:
    """Namespace for sandbox operations on a `TaskEnvironment`.

    Accessed via `env.sandbox`.  Provides a unified `orchestrator()`
    method that acts as a decorator (when given a callable), a code-string
    task factory (when given a string), or a decorator factory (when given
    only keyword arguments).
    """

    def __init__(self, env: TaskEnvironment) -> None:
        self._env = env

    @overload
    def orchestrator(
        self,
        _func_or_source: Callable,
        /,
    ) -> "SandboxedTaskTemplate": ...

    @overload
    def orchestrator(
        self,
        _func_or_source: str,
        /,
        *,
        tasks: list[Any] | None = None,
        inputs: dict[str, type] | None = None,
        output: type = type(None),
        name: str | None = None,
        timeout_ms: int = 30_000,
        cache: CacheRequest | None = None,
        retries: int = 0,
    ) -> "CodeTaskTemplate": ...

    @overload
    def orchestrator(
        self,
        *,
        timeout_ms: int = 30_000,
        max_memory: int = 50 * 1024 * 1024,
        max_stack_depth: int = 256,
        type_check: bool = True,
        name: str | None = None,
        cache: CacheRequest | None = None,
        retries: int = 0,
    ) -> "Callable[[Callable], SandboxedTaskTemplate]": ...

    def orchestrator(  # type: ignore[misc]
        self,
        _func_or_source: Any = None,
        /,
        *,
        tasks: list[Any] | None = None,
        inputs: dict[str, type] | None = None,
        output: type = type(None),
        timeout_ms: int = 30_000,
        max_memory: int = 50 * 1024 * 1024,
        max_stack_depth: int = 256,
        type_check: bool = True,
        name: str | None = None,
        cache: CacheRequest | None = None,
        retries: int = 0,
    ) -> Any:
        """Unified sandbox orchestration on a `TaskEnvironment`.

        Three usage modes:

        1. **Decorator** (callable) — creates a `SandboxedTaskTemplate`::

            @env.sandbox.orchestrator
            def pipeline(n: int) -> dict: ...

        2. **Code string** — creates a `CodeTaskTemplate`::

            task = env.sandbox.orchestrator(
                "add(x, y) * 2",
                tasks=[add],
                inputs={"x": int},
                output=int,
            )

        3. **Decorator factory** (keyword-only) — returns a decorator::

            @env.sandbox.orchestrator(timeout_ms=5000)
            def pipeline(n: int) -> dict: ...
        """
        from .sandbox._config import SandboxedConfig
        from .sandbox._task import SandboxedTaskTemplate

        env = self._env

        if _func_or_source is None:
            # Mode 3: decorator factory — keyword-only args
            config = SandboxedConfig(
                max_memory=max_memory,
                max_stack_depth=max_stack_depth,
                timeout_ms=timeout_ms,
                type_check=type_check,
            )

            def decorator(func: Callable) -> SandboxedTaskTemplate:
                task_name = name or (env.name + "." + func.__name__)
                interface = NativeInterface.from_callable(func)
                tmpl = SandboxedTaskTemplate(
                    func=func,
                    name=task_name,
                    interface=interface,
                    plugin_config=config,
                    image=env.image,
                    cache=cache or env.cache,
                    retries=retries,
                    parent_env=weakref.ref(env),
                    parent_env_name=env.name,
                )
                env._tasks[task_name] = tmpl
                return tmpl

            return decorator

        if callable(_func_or_source) and not isinstance(_func_or_source, str):
            # Mode 1: bare decorator
            func = _func_or_source
            config = SandboxedConfig(
                max_memory=max_memory,
                max_stack_depth=max_stack_depth,
                timeout_ms=timeout_ms,
                type_check=type_check,
            )
            task_name = name or (env.name + "." + func.__name__)
            interface = NativeInterface.from_callable(func)
            tmpl = SandboxedTaskTemplate(
                func=func,
                name=task_name,
                interface=interface,
                plugin_config=config,
                image=env.image,
                cache=cache or env.cache,
                retries=retries,
                parent_env=weakref.ref(env),
                parent_env_name=env.name,
            )
            env._tasks[task_name] = tmpl
            return tmpl

        if isinstance(_func_or_source, str):
            # Mode 2: code string
            import sys

            from .sandbox._api import _orchestrator_impl

            return _orchestrator_impl(
                _func_or_source,
                inputs=inputs or {},
                output=output,
                tasks=tasks,
                name=name or "sandboxed-code",
                timeout_ms=timeout_ms,
                cache=cache or env.cache,
                retries=retries,
                image=env.image,
                caller_module=sys._getframe(1).f_globals.get("__name__", "__main__"),
            )

        raise TypeError(f"orchestrator() expects a callable, string, or keyword arguments, got {type(_func_or_source)}")
