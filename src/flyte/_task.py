from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    ParamSpec,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from flyte._pod import PodTemplate
from flyte.errors import RuntimeSystemError, RuntimeUserError

from ._cache import Cache, CacheRequest
from ._context import internal_ctx
from ._doc import Documentation
from ._image import Image
from ._resources import Resources
from ._retry import RetryStrategy
from ._reusable_environment import ReusePolicy
from ._secret import SecretRequest
from ._timeout import TimeoutType
from .models import NativeInterface, SerializationContext

if TYPE_CHECKING:
    from flyteidl.core.tasks_pb2 import DataLoadingConfig

    from ._task_environment import TaskEnvironment

P = ParamSpec("P")  # capture the function's parameters
R = TypeVar("R")  # return type

AsyncFunctionType: TypeAlias = Callable[P, Coroutine[Any, Any, R]]
SyncFunctionType: TypeAlias = Callable[P, R]
FunctionTypes: TypeAlias = Union[AsyncFunctionType, SyncFunctionType]


@dataclass(kw_only=True)
class TaskTemplate(Generic[P, R]):
    """
    Task template is a template for a task that can be executed. It defines various parameters for the task, which
    can be defined statically at the time of task definition or dynamically at the time of task invocation using
    the override method.

    Example usage:
    ```python
    @task(name="my_task", image="my_image", resources=Resources(cpu="1", memory="1Gi"))
    def my_task():
        pass
    ```

    :param name: Optional The name of the task (defaults to the function name)
    :param task_type: Router type for the task, this is used to determine how the task will be executed.
     This is usually set to match with th execution plugin.
    :param image: Optional The image to use for the task, if set to "auto" will use the default image for the python
    version with flyte installed
    :param resources: Optional The resources to use for the task
    :param cache: Optional The cache policy for the task, defaults to auto, which will cache the results of the task.
    :param interruptable: Optional The interruptable policy for the task, defaults to False, which means the task
     will not be scheduled on interruptable nodes. If set to True, the task will be scheduled on interruptable nodes,
     and the code should handle interruptions and resumptions.
    :param retries: Optional The number of retries for the task, defaults to 0, which means no retries.
    :param reusable: Optional The reusability policy for the task, defaults to None, which means the task environment
    will not be reused across task invocations.
    :param docs: Optional The documentation for the task, if not provided the function docstring will be used.
    :param env: Optional The environment variables to set for the task.
    :param secrets: Optional The secrets that will be injected into the task at runtime.
    :param timeout: Optional The timeout for the task.
    """

    name: str
    interface: NativeInterface
    friendly_name: str = ""
    task_type: str = "python"
    task_type_version: int = 0
    image: Union[str, Image, Literal["auto"]] = "auto"
    resources: Optional[Resources] = None
    cache: CacheRequest = "auto"
    interruptable: bool = False
    retries: Union[int, RetryStrategy] = 0
    reusable: Union[ReusePolicy, None] = None
    docs: Optional[Documentation] = None
    env: Optional[Dict[str, str]] = None
    secrets: Optional[SecretRequest] = None
    timeout: Optional[TimeoutType] = None
    pod_template: Optional[Union[str, PodTemplate]] = None
    report: bool = False

    parent_env: Optional[weakref.ReferenceType[TaskEnvironment]] = None
    local: bool = field(default=False, init=False)
    ref: bool = field(default=False, init=False, repr=False, compare=False)

    # Only used in python 3.10 and 3.11, where we cannot use markcoroutinefunction
    _call_as_synchronous: bool = False

    def __post_init__(self):
        # Auto set the image based on the image request
        if self.image == "auto":
            self.image = Image.from_debian_base()
        elif isinstance(self.image, str):
            self.image = Image.from_base(str(self.image))

        # Auto set cache based on the cache request
        if isinstance(self.cache, str):
            match self.cache:
                case "auto":
                    self.cache = Cache(behavior="auto")
                case "override":
                    self.cache = Cache(behavior="override")
                case "disable":
                    self.cache = Cache(behavior="disable")

        # if retries is set to int, convert to RetryStrategy
        if isinstance(self.retries, int):
            self.retries = RetryStrategy(count=self.retries)

        if self.friendly_name == "":
            # If friendly_name is not set, use the name of the task
            self.friendly_name = self.name

    def __getstate__(self):
        """
        This method is called when the object is pickled. We need to remove the parent_env reference
        to avoid circular references.
        """
        state = self.__dict__.copy()
        state.pop("parent_env", None)
        return state

    def __setstate__(self, state):
        """
        This method is called when the object is unpickled. We need to set the parent_env reference
        to the environment that created the task.
        """
        self.__dict__.update(state)
        self.parent_env = None

    @property
    def source_file(self) -> Optional[str]:
        return None

    async def pre(self, *args, **kwargs) -> Dict[str, Any]:
        """
        This is the preexecute function that will be
        called before the task is executed
        """
        return {}

    async def execute(self, *args, **kwargs) -> Any:
        """
        This is the pure python function that will be executed when the task is called.
        """
        raise NotImplementedError

    async def post(self, return_vals: Any) -> Any:
        """
        This is the postexecute function that will be
        called after the task is executed
        """
        return return_vals

    # ---- Extension points ----
    def config(self, sctx: SerializationContext) -> Dict[str, str]:
        """
        Returns additional configuration for the task. This is a set of key-value pairs that can be used to
        configure the task execution environment at runtime. This is usually used by plugins.
        """
        return {}

    def custom_config(self, sctx: SerializationContext) -> Dict[str, str]:
        """
        Returns additional configuration for the task. This is a set of key-value pairs that can be used to
        configure the task execution environment at runtime. This is usually used by plugins.
        """
        return {}

    def data_loading_config(self, sctx: SerializationContext) -> DataLoadingConfig:
        """
        This configuration allows executing raw containers in Flyte using the Flyte CoPilot system
        Flyte CoPilot, eliminates the needs of sdk inside the container. Any inputs required by the users container
        are side-loaded in the input_path
        Any outputs generated by the user container - within output_path are automatically uploaded
        """

    def container_args(self, sctx: SerializationContext) -> List[str]:
        """
        Returns the container args for the task. This is a set of key-value pairs that can be used to
        configure the task execution environment at runtime. This is usually used by plugins.
        """
        return []

    def sql(self, sctx: SerializationContext) -> Optional[str]:
        """
        Returns the SQL for the task. This is a set of key-value pairs that can be used to
        configure the task execution environment at runtime. This is usually used by plugins.
        """
        return None

    # ---- Extension points ----

    @property
    def native_interface(self) -> NativeInterface:
        return self.interface

    async def aio(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R] | R:
        """
        The aio function allows executing "sync" tasks, in an async context. This helps with migrating v1 defined sync
        tasks to be used within an asyncio parent task.
        This function will also re-raise exceptions from the underlying task.

        Example:
        ```python
        @env.task
        def my_legacy_task(x: int) -> int:
            return x

        @env.task
        async def my_new_parent_task(n: int) -> List[int]:
            collect = []
            for x in range(n):
                collect.append(my_legacy_task.aio(x))
            return asyncio.gather(*collect)
        ```
        :param args:
        :param kwargs:
        :return:
        """

        ctx = internal_ctx()
        if ctx.is_task_context():
            from ._internal.controllers import get_controller

            # If we are in a task context, that implies we are executing a Run.
            # In this scenario, we should submit the task to the controller.
            controller = get_controller()
            if controller:
                if self._call_as_synchronous:
                    fut = controller.submit_sync(self, *args, **kwargs)
                    asyncio_future = asyncio.wrap_future(fut)  # Wrap the future to make it awaitable
                    return await asyncio_future
                else:
                    return await controller.submit(self, *args, **kwargs)
            else:
                raise RuntimeSystemError("BadContext", "Controller is not initialized.")
        else:
            # Local execute, just stay out of the way, but because .aio is used, we want to return an awaitable,
            # even for synchronous tasks. This is to support migration.
            return self.forward(*args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R] | R:
        """
        This is the entrypoint for an async function task at runtime. It will be called during an execution.
        Please do not override this method, if you simply want to modify the execution behavior, override the
        execute method.

        This needs to be overridable to maybe be async.
        The returned thing from here needs to be an awaitable if the underlying task is async, and a regular object
        if the task is not.
        """
        try:
            ctx = internal_ctx()
            if ctx.is_task_context():
                # If we are in a task context, that implies we are executing a Run.
                # In this scenario, we should submit the task to the controller.
                # We will also check if we are not initialized, It is not expected to be not initialized
                from ._internal.controllers import get_controller

                controller = get_controller()
                if not controller:
                    raise RuntimeSystemError("BadContext", "Controller is not initialized.")

                if self._call_as_synchronous:
                    fut = controller.submit_sync(self, *args, **kwargs)
                    x = fut.result(None)
                    return x
                else:
                    return controller.submit(self, *args, **kwargs)
            else:
                # If not in task context, purely function run, stay out of the way
                return self.forward(*args, **kwargs)
        except RuntimeSystemError:
            raise
        except RuntimeUserError:
            raise
        except Exception as e:
            raise RuntimeUserError(type(e).__name__, str(e)) from e

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R] | R:
        """
        Think of this as a local execute method for your task. This function will be invoked by the __call__ method
        when not in a Flyte task execution context.  See the implementation below for an example.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def override(
        self,
        *,
        resources: Optional[Resources] = None,
        cache: CacheRequest = "auto",
        retries: Union[int, RetryStrategy] = 0,
        timeout: Optional[TimeoutType] = None,
        reusable: Union[ReusePolicy, Literal["off"], None] = None,
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        **kwargs: Any,
    ) -> TaskTemplate:
        """
        Override various parameters of the task template. This allows for dynamic configuration of the task
        when it is called, such as changing the image, resources, cache policy, etc.
        """
        cache = cache or self.cache
        retries = retries or self.retries
        timeout = timeout or self.timeout
        reusable = reusable or self.reusable
        if reusable == "off":
            reusable = None

        if reusable is not None:
            if resources is not None:
                raise ValueError(
                    "Cannot override resources when reusable is set."
                    " Reusable tasks will use the parent env's resources. You can disable reusability and"
                    " override resources if needed. (set reusable='off')"
                )
            if env is not None:
                raise ValueError(
                    "Cannot override env when reusable is set."
                    " Reusable tasks will use the parent env's env. You can disable reusability and "
                    "override env if needed. (set reusable='off')"
                )
            if secrets is not None:
                raise ValueError(
                    "Cannot override secrets when reusable is set."
                    " Reusable tasks will use the parent env's secrets. You can disable reusability and "
                    "override secrets if needed. (set reusable='off')"
                )

        resources = resources or self.resources
        env = env or self.env
        secrets = secrets or self.secrets

        for k, v in kwargs.items():
            if k == "name":
                raise ValueError("Name cannot be overridden")
            if k == "image":
                raise ValueError("Image cannot be overridden")
            if k == "docs":
                raise ValueError("Docs cannot be overridden")
            if k == "interface":
                raise ValueError("Interface cannot be overridden")

        return replace(
            self,
            resources=resources,
            cache=cache,
            retries=retries,
            timeout=timeout,
            reusable=cast(Optional[ReusePolicy], reusable),
            env=env,
            secrets=secrets,
        )


@dataclass(kw_only=True)
class AsyncFunctionTaskTemplate(TaskTemplate[P, R]):
    """
    A task template that wraps an asynchronous functions. This is automatically created when an asynchronous function
    is decorated with the task decorator.
    """

    func: FunctionTypes
    plugin_config: Optional[Any] = None  # This is used to pass plugin specific configuration

    def __post_init__(self):
        super().__post_init__()
        if not iscoroutinefunction(self.func):
            self._call_as_synchronous = True

    @property
    def source_file(self) -> Optional[str]:
        """
        Returns the source file of the function, if available. This is useful for debugging and tracing.
        """
        if hasattr(self.func, "__code__") and self.func.__code__:
            return self.func.__code__.co_filename
        return None

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R] | R:
        # In local execution, we want to just call the function. Note we're not awaiting anything here.
        # If the function was a coroutine function, the coroutine is returned and the await that the caller has
        # in front of the task invocation will handle the awaiting.
        return self.func(*args, **kwargs)

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        This is the execute method that will be called when the task is invoked. It will call the actual function.
        # TODO We may need to keep this as the bare func execute, and need a pre and post execute some other func.
        """

        ctx = internal_ctx()
        assert ctx.data.task_context is not None, "Function should have already returned if not in a task context"
        ctx_data = await self.pre(*args, **kwargs)
        tctx = ctx.data.task_context.replace(data=ctx_data)
        with ctx.replace_task_context(tctx):
            if iscoroutinefunction(self.func):
                v = await self.func(*args, **kwargs)
            else:
                v = self.func(*args, **kwargs)
            await self.post(v)
        return v

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        args = [
            "a0",
            "--inputs",
            serialize_context.input_path,
            "--outputs-path",
            serialize_context.output_path,
            "--version",
            serialize_context.version,  # pr: should this be serialize_context.version or code_bundle.version?
            "--raw-data-path",
            "{{.rawOutputDataPrefix}}",
            "--checkpoint-path",
            "{{.checkpointOutputPrefix}}",
            "--prev-checkpoint",
            "{{.prevCheckpointPrefix}}",
            "--run-name",
            "{{.runName}}",
            "--name",
            "{{.actionName}}",
        ]
        # Add on all the known images
        if serialize_context.image_cache and serialize_context.image_cache.serialized_form:
            args = [*args, "--image-cache", serialize_context.image_cache.serialized_form]
        else:
            if serialize_context.image_cache:
                args = [*args, "--image-cache", serialize_context.image_cache.to_transport]

        if serialize_context.code_bundle:
            if serialize_context.code_bundle.tgz:
                args = [*args, *["--tgz", f"{serialize_context.code_bundle.tgz}"]]
            elif serialize_context.code_bundle.pkl:
                args = [*args, *["--pkl", f"{serialize_context.code_bundle.pkl}"]]
            args = [*args, *["--dest", f"{serialize_context.code_bundle.destination or '.'}"]]

        if not serialize_context.code_bundle or not serialize_context.code_bundle.pkl:
            # If we do not have a code bundle, or if we have one, but it is not a pkl, we need to add the resolver

            from flyte._internal.resolvers.default import DefaultTaskResolver

            _task_resolver = DefaultTaskResolver()
            args = [
                *args,
                *[
                    "--resolver",
                    _task_resolver.import_path,
                    *_task_resolver.loader_args(task=self, root_dir=serialize_context.root_dir),
                ],
            ]

        assert all(isinstance(item, str) for item in args), f"All args should be strings, non string item = {args}"

        return args
