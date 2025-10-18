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


@rich.repr.auto
@dataclass(init=True, repr=True)
class TaskEnvironment(Environment):
    """
    Environment class to define a new environment for a set of tasks.

    Example usage:
    ```python
    env = flyte.TaskEnvironment(name="my_env", image="my_image", resources=Resources(cpu="1", memory="1Gi"))

    @env.task
    async def my_task():
        pass
    ```

    :param name: Name of the environment
    :param image: Docker image to use for the environment. If set to "auto", will use the default image.
    :param resources: Resources to allocate for the environment.
    :param env_vars: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param depends_on: Environment dependencies to hint, so when you deploy the environment,
        the dependencies are also deployed. This is useful when you have a set of environments
        that depend on each other.
    :param cache: Cache policy for the environment.
    :param reusable: Reuse policy for the environment, if set, a python process may be reused for multiple tasks.
    :param plugin_config: Optional plugin configuration for custom task types.
        If set, all tasks in this environment will use the specified plugin configuration.
    :param queue: Optional queue name to use for tasks in this environment.
        If not set, the default queue will be used.
    :param pod_template: Optional pod template to use for tasks in this environment.
        If not set, the default pod template will be used.
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
        **kwargs: Any,
    ) -> TaskEnvironment:
        """
        Clone the TaskEnvironment with new parameters.

        Besides the base environment parameters, you can override kwargs like `cache`, `reusable`, etc.

        :param name: The name of the environment.
        :param image: The image to use for the environment.
        :param resources: The resources to allocate for the environment.
        :param env_vars: The environment variables to set for the environment.
        :param secrets: The secrets to inject into the environment.
        :param depends_on: The environment dependencies to hint, so when you deploy the environment,
            the dependencies are also deployed. This is useful when you have a set of environments
            that depend on each other.
        :param queue: The queue name to use for tasks in this environment.
        :param pod_template: The pod template to use for tasks in this environment.
        :param description: The description of the environment.
        :param interruptible: Whether the environment is interruptible and can be scheduled on spot/preemptible
            instances.
        :param kwargs: Additional parameters to override the environment (e.g., cache, reusable, plugin_config).
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
    ) -> Callable[[F], AsyncFunctionTaskTemplate[P, R, F]] | AsyncFunctionTaskTemplate[P, R, F]:
        """
        Decorate a function to be a task.

        :param _func: Optional The function to decorate. If not provided, the decorator will return a callable that
        accepts a function to be decorated.
        :param short_name: Optional A friendly name for the task (defaults to the function name)
        :param cache: Optional The cache policy for the task, defaults to auto, which will cache the results of the
        task.
        :param retries: Optional The number of retries for the task, defaults to 0, which means no retries.
        :param docs: Optional The documentation for the task, if not provided the function docstring will be used.
        :param timeout: Optional The timeout for the task.
        :param pod_template: Optional The pod template for the task, if not provided the default pod template will be
        used.
        :param report: Optional Whether to generate the html report for the task, defaults to False.
        :param max_inline_io_bytes: Maximum allowed size (in bytes) for all inputs and outputs passed directly to the
         task (e.g., primitives, strings, dicts). Does not apply to files, directories, or dataframes.
        :param triggers: Optional A tuple of triggers to associate with the task. This allows the task to be run on a
         schedule or in response to events. Triggers can be defined using the `flyte.trigger` module.
        :param interruptible: Optional Whether the task is interruptible, defaults to environment setting.
        :param queue: Optional queue name to use for this task. If not set, the environment's queue will be used.

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
                triggers=triggers if isinstance(triggers, tuple) else (triggers,),
            )
            self._tasks[task_name] = tmpl
            return tmpl

        if _func is None:
            return cast(Callable[[F], AsyncFunctionTaskTemplate[P, R, F]], decorator)
        return cast(AsyncFunctionTaskTemplate[P, R, F], decorator(_func))

    @property
    def tasks(self) -> Dict[str, TaskTemplate]:
        """
        Get all tasks defined in the environment.
        """
        return self._tasks

    @classmethod
    def from_task(cls, name: str, *tasks: TaskTemplate) -> TaskEnvironment:
        """
        Create a TaskEnvironment from a list of tasks. All tasks should have the same image or no Image defined.
        Similarity of Image is determined by the python reference, not by value.

        If images are different, an error is raised. If no image is defined, the image is set to "auto".

        For any other tasks that need to be use these tasks, the returned environment can be used in the `depends_on`
        attribute of the other TaskEnvironment.

        :param name: The name of the environment.
        :param tasks: The list of tasks to create the environment from.

        :raises ValueError: If tasks are assigned to multiple environments or have different images.
        :return: The created TaskEnvironment.
        """
        envs = [t.parent_env() for t in tasks if t.parent_env and t.parent_env() is not None]
        if envs:
            raise ValueError("Tasks cannot assigned to multiple environments.")
        images = {t.image for t in tasks}
        if len(images) > 1:
            raise ValueError("Tasks must have the same image to be in the same environment.")
        image: Union[str, Image] = images.pop() if images else "auto"
        env = cls(name, image=image)
        for t in tasks:
            env._tasks[t.name] = t
            t.parent_env = weakref.ref(env)
            t.parent_env_name = name
        return env
