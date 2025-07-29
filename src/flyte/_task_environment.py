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
    Union,
    cast,
)

import rich.repr

from ._cache import CacheRequest
from ._doc import Documentation
from ._environment import Environment
from ._image import Image
from ._resources import Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._secret import SecretRequest
from ._task import AsyncFunctionTaskTemplate, TaskTemplate
from .models import NativeInterface

if TYPE_CHECKING:
    from kubernetes.client import V1PodTemplate

    from ._task import FunctionTypes, P, R


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
    :param env: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param depends_on: Environment dependencies to hint, so when you deploy the environment, the dependencies are
        also deployed. This is useful when you have a set of environments that depend on each other.
    :param cache: Cache policy for the environment.
    :param reusable: Reuse policy for the environment, if set, a python process may be reused for multiple tasks.
    """

    cache: Union[CacheRequest] = "disable"
    reusable: ReusePolicy | None = None
    plugin_config: Optional[Any] = None
    # TODO Shall we make this union of string or env? This way we can lookup the env by module/file:name
    # TODO also we could add list of files that are used by this environment

    _tasks: Dict[str, TaskTemplate] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reusable is not None and self.plugin_config is not None:
            raise ValueError("Cannot set plugin_config when environment is reusable.")

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[List[Environment]] = None,
        **kwargs: Any,
    ) -> TaskEnvironment:
        """
        Clone the TaskEnvironment with new parameters.
        besides the base environment parameters, you can override, kwargs like `cache`, `reusable`, etc.

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
        if env is not None:
            kwargs["env"] = env
        if reusable_set:
            kwargs["reusable"] = reusable
        if secrets is not None:
            kwargs["secrets"] = secrets
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        return replace(self, **kwargs)

    def task(
        self,
        _func=None,
        *,
        name: Optional[str] = None,
        cache: Union[CacheRequest] | None = None,
        retries: Union[int, RetryStrategy] = 0,
        timeout: Union[timedelta, int] = 0,
        docs: Optional[Documentation] = None,
        secrets: Optional[SecretRequest] = None,
        pod_template: Optional[Union[str, "V1PodTemplate"]] = None,
        report: bool = False,
    ) -> Union[AsyncFunctionTaskTemplate, Callable[P, R]]:
        """
        :param _func: Optional The function to decorate. If not provided, the decorator will return a callable that
        :param name: Optional A friendly name for the task (defaults to the function name)
        :param cache: Optional The cache policy for the task, defaults to auto, which will cache the results of the
        task.
        :param retries: Optional The number of retries for the task, defaults to 0, which means no retries.
        :param docs: Optional The documentation for the task, if not provided the function docstring will be used.
        :param secrets: Optional The secrets that will be injected into the task at runtime.
        :param timeout: Optional The timeout for the task.
        :param pod_template: Optional The pod template for the task, if not provided the default pod template will be
        used.
        :param report: Optional Whether to generate the html report for the task, defaults to False.
        """
        from ._task import P, R

        if self.reusable is not None:
            if pod_template is not None:
                raise ValueError("Cannot set pod_template when environment is reusable.")

        def decorator(func: FunctionTypes) -> AsyncFunctionTaskTemplate[P, R]:
            friendly_name = name or func.__name__
            task_name = self.name + "." + func.__name__

            if not inspect.iscoroutinefunction(func) and self.reusable is not None:
                if self.reusable.concurrency > 1:
                    raise ValueError(
                        "Reusable environments with concurrency greater than 1 are only supported for async tasks. "
                        "Please use an async function or set concurrency to 1."
                    )

            if self.plugin_config is not None:
                from flyte.extend import TaskPluginRegistry

                task_template_class: type[AsyncFunctionTaskTemplate[P, R]] | None = TaskPluginRegistry.find(
                    config_type=type(self.plugin_config)
                )
                if task_template_class is None:
                    raise ValueError(
                        f"No task plugin found for config type {type(self.plugin_config)}. "
                        f"Please register a plugin using flyte.extend.TaskPluginRegistry.register() api."
                    )
            else:
                task_template_class = AsyncFunctionTaskTemplate[P, R]

            task_template_class = cast(type[AsyncFunctionTaskTemplate[P, R]], task_template_class)
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
                env=self.env,
                secrets=secrets or self.secrets,
                pod_template=pod_template or self.pod_template,
                parent_env=weakref.ref(self),
                interface=NativeInterface.from_callable(func),
                report=report,
                friendly_name=friendly_name,
                plugin_config=self.plugin_config,
            )
            self._tasks[task_name] = tmpl
            return tmpl

        if _func is None:
            return cast(AsyncFunctionTaskTemplate, decorator)
        return cast(AsyncFunctionTaskTemplate, decorator(_func))

    @property
    def tasks(self) -> Dict[str, TaskTemplate]:
        """
        Get all tasks defined in the environment.
        """
        return self._tasks

    def add_task(self, task: TaskTemplate) -> TaskTemplate:
        """
        Add a task to the environment.
        """
        if task.name in self._tasks:
            raise ValueError(f"Task {task.name} already exists in the environment. Task names should be unique.")
        self._tasks[task.name] = task
        task.parent_env = weakref.ref(self)
        return task
