from __future__ import annotations

import functools
from dataclasses import dataclass
from threading import Lock
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, Iterator, Literal, Optional, Tuple, Union

import rich.repr
from google.protobuf import timestamp

import flyte
import flyte.errors
from flyte._context import internal_ctx
from flyte._initialize import ensure_client, get_client, get_common_config
from flyte._logging import logger
from flyte._protos.common import identifier_pb2, list_pb2
from flyte._protos.workflow import task_definition_pb2, task_service_pb2
from flyte.models import NativeInterface
from flyte.syncify import syncify


def _repr_task_metadata(metadata: task_definition_pb2.TaskMetadata) -> rich.repr.Result:
    """
    Rich representation of the task metadata.
    """
    if metadata.deployed_by:
        if metadata.deployed_by.user:
            yield "deployed_by", f"User: {metadata.deployed_by.user.spec.email}"
        else:
            yield "deployed_by", f"App: {metadata.deployed_by.application.spec.name}"
    yield "short_name", metadata.short_name
    yield "deployed_at", timestamp.to_datetime(metadata.deployed_at)
    yield "environment_name", metadata.environment_name


class LazyEntity:
    """
    Fetches the entity when the entity is called or when the entity is retrieved.
    The entity is derived from RemoteEntity so that it behaves exactly like the mimicked entity.
    """

    def __init__(self, name: str, getter: Callable[..., Coroutine[Any, Any, TaskDetails]], *args, **kwargs):
        self._task: Optional[TaskDetails] = None
        self._getter = getter
        self._name = name
        self._mutex = Lock()

    @property
    def name(self) -> str:
        return self._name

    @syncify
    async def fetch(self) -> TaskDetails:
        """
        Forwards all other attributes to task, causing the task to be fetched!
        """
        with self._mutex:
            if self._task is None:
                self._task = await self._getter()
            if self._task is None:
                raise RuntimeError(f"Error downloading the task {self._name}, (check original exception...)")
            return self._task

    async def __call__(self, *args, **kwargs):
        """
        Forwards the call to the underlying task. The entity will be fetched if not already present
        """
        tk = await self.fetch.aio()
        return await tk(*args, **kwargs)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Future for task with name {self._name}"


AutoVersioning = Literal["latest", "current"]


@dataclass
class TaskDetails:
    pb2: task_definition_pb2.TaskDetails

    @classmethod
    def get(
        cls,
        name: str,
        project: str | None,
        domain: str | None,
        version: str | None = None,
        auto_version: AutoVersioning | None = None,
    ) -> LazyEntity:
        """
        Get a task by its ID or name. If both are provided, the ID will take precedence.

        Either version or auto_version are required parameters.

        :param name: The name of the task.
        :param project: The project of the task.
        :param domain: The domain of the task.
        :param version: The version of the task.
        :param auto_version: If set to "latest", the latest-by-time ordered from now, version of the task will be used.
         If set to "current", the version will be derived from the callee tasks context. This is useful if you are
         deploying all environments with the same version. If auto_version is current, you can only access the task from
         within a task context.
        """

        if version is None and auto_version is None:
            raise ValueError("Either version or auto_version must be provided.")

        if version is None and auto_version not in ["latest", "current"]:
            raise ValueError("auto_version must be either 'latest' or 'current'.")

        async def deferred_get(_version: str | None, _auto_version: AutoVersioning | None) -> TaskDetails:
            if _version is None:
                if _auto_version == "latest":
                    tasks = []
                    async for x in Task.listall.aio(
                        by_task_name=name,
                        project=project,
                        domain=domain,
                        sort_by=("created_at", "desc"),
                        limit=1,
                    ):
                        tasks.append(x)
                    if not tasks:
                        raise flyte.errors.ReferenceTaskError(f"Task {name} not found.")
                    _version = tasks[0].version
                elif _auto_version == "current":
                    ctx = flyte.ctx()
                    if ctx is None:
                        raise ValueError("auto_version=current can only be used within a task context.")
                    _version = ctx.version
            cfg = get_common_config()
            task_id = task_definition_pb2.TaskIdentifier(
                org=cfg.org,
                project=project or cfg.project,
                domain=domain or cfg.domain,
                name=name,
                version=_version,
            )
            resp = await get_client().task_service.GetTaskDetails(
                task_service_pb2.GetTaskDetailsRequest(
                    task_id=task_id,
                )
            )
            return cls(resp.details)

        return LazyEntity(
            name=name, getter=functools.partial(deferred_get, _version=version, _auto_version=auto_version)
        )

    @property
    def name(self) -> str:
        """
        The name of the task.
        """
        return self.pb2.task_id.name

    @property
    def version(self) -> str:
        """
        The version of the task.
        """
        return self.pb2.task_id.version

    @property
    def task_type(self) -> str:
        """
        The type of the task.
        """
        return self.pb2.spec.task_template.type

    @property
    def default_input_args(self) -> Tuple[str, ...]:
        """
        The default input arguments of the task.
        """
        return tuple(x.name for x in self.pb2.spec.default_inputs)

    @property
    def required_args(self) -> Tuple[str, ...]:
        """
        The required input arguments of the task.
        """
        return tuple(x for x, _ in self.interface.inputs.items() if x not in self.default_input_args)

    @functools.cached_property
    def interface(self) -> NativeInterface:
        """
        The interface of the task.
        """
        import flyte.types as types

        return types.guess_interface(self.pb2.spec.task_template.interface, default_inputs=self.pb2.spec.default_inputs)

    @property
    def cache(self) -> flyte.Cache:
        """
        The cache policy of the task.
        """
        return flyte.Cache(
            behavior="enabled" if self.pb2.spec.task_template.metadata.discoverable else "disable",
            version_override=self.pb2.spec.task_template.metadata.discovery_version,
            serialize=self.pb2.spec.task_template.metadata.cache_serializable,
            ignored_inputs=tuple(self.pb2.spec.task_template.metadata.cache_ignore_input_vars),
        )

    @property
    def secrets(self):
        """
        The secrets of the task.
        """
        return [s.key for s in self.pb2.spec.task_template.security_context.secrets]

    @property
    def resources(self):
        """
        The resources of the task.
        """
        if self.pb2.spec.task_template.container is None:
            return ()
        return (
            self.pb2.spec.task_template.container.resources.requests,
            self.pb2.spec.task_template.container.resources.limits,
        )

    async def __call__(self, *args, **kwargs):
        """
        Forwards the call to the underlying task. The entity will be fetched if not already present
        """
        # TODO support kwargs, for this we need ordered inputs to be stored in the task spec.
        if len(args) > 0:
            raise flyte.errors.ReferenceTaskError(
                f"Reference task {self.name} does not support positional arguments"
                f"currently. Please use keyword arguments."
            )
        if len(self.required_args) > 0:
            if len(args) + len(kwargs) < len(self.required_args):
                raise ValueError(
                    f"Task {self.name} requires at least {self.required_args} arguments, "
                    f"but only received args:{args}  kwargs{kwargs}."
                )

        ctx = internal_ctx()
        if ctx.is_task_context():
            # If we are in a task context, that implies we are executing a Run.
            # In this scenario, we should submit the task to the controller.
            # We will also check if we are not initialized, It is not expected to be not initialized
            from flyte._internal.controllers import get_controller

            controller = get_controller()
            if controller:
                return await controller.submit_task_ref(self.pb2, *args, **kwargs)
        raise flyte.errors

    def override(
        self,
        *,
        local: Optional[bool] = None,
        ref: Optional[bool] = None,
        resources: Optional[flyte.Resources] = None,
        cache: flyte.CacheRequest = "auto",
        retries: Union[int, flyte.RetryStrategy] = 0,
        timeout: Optional[flyte.TimeoutType] = None,
        reusable: Union[flyte.ReusePolicy, Literal["auto"], None] = None,
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[flyte.SecretRequest] = None,
        **kwargs: Any,
    ) -> TaskDetails:
        raise NotImplementedError

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the task.
        """
        yield "friendly_name", self.pb2.spec.short_name
        yield "environment", self.pb2.spec.environment
        yield "default_inputs_keys", self.default_input_args
        yield "required_args", self.required_args
        yield "raw_default_inputs", [str(x) for x in self.pb2.spec.default_inputs]
        yield "project", self.pb2.task_id.project
        yield "domain", self.pb2.task_id.domain
        yield "name", self.name
        yield "version", self.version
        yield "task_type", self.task_type
        yield "cache", self.cache
        yield "interface", self.name + str(self.interface)
        yield "secrets", self.secrets
        yield "resources", self.resources


@dataclass
class Task:
    pb2: task_definition_pb2.Task

    def __init__(self, pb2: task_definition_pb2.Task):
        self.pb2 = pb2

    @property
    def name(self) -> str:
        """
        The name of the task.
        """
        return self.pb2.task_id.name

    @property
    def version(self) -> str:
        """
        The version of the task.
        """
        return self.pb2.task_id.version

    @classmethod
    def get(
        cls,
        name: str,
        project: str | None = None,
        domain: str | None = None,
        version: str | None = None,
        auto_version: AutoVersioning | None = None,
    ) -> LazyEntity:
        """
        Get a task by its ID or name. If both are provided, the ID will take precedence.

        Either version or auto_version are required parameters.

        :param name: The name of the task.
        :param project: The project of the task.
        :param domain: The domain of the task.
        :param version: The version of the task.
        :param auto_version: If set to "latest", the latest-by-time ordered from now, version of the task will be used.
         If set to "current", the version will be derived from the callee tasks context. This is useful if you are
         deploying all environments with the same version. If auto_version is current, you can only access the task from
         within a task context.
        """
        return TaskDetails.get(name, project=project, domain=domain, version=version, auto_version=auto_version)

    @syncify
    @classmethod
    async def listall(
        cls,
        by_task_name: str | None = None,
        project: str | None = None,
        domain: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
        limit: int = 100,
    ) -> Union[AsyncIterator[Task], Iterator[Task]]:
        """
        Get all runs for the current project and domain.

        :param by_task_name: If provided, only tasks with this name will be returned.
        :param project: The project to filter tasks by. If None, the current project will be used.
        :param domain: The domain to filter tasks by. If None, the current domain will be used.
        :param sort_by: The sorting criteria for the project list, in the format (field, order).
        :param limit: The maximum number of tasks to return.
        :return: An iterator of runs.
        """
        ensure_client()
        token = None
        sort_by = sort_by or ("created_at", "asc")
        sort_pb2 = list_pb2.Sort(
            key=sort_by[0], direction=list_pb2.Sort.ASCENDING if sort_by[1] == "asc" else list_pb2.Sort.DESCENDING
        )
        cfg = get_common_config()
        filters = []
        if by_task_name:
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.Function.EQUAL,
                    field="name",
                    values=[by_task_name],
                )
            )
        original_limit = limit
        if limit > cfg.batch_size:
            limit = cfg.batch_size
        retrieved = 0
        while True:
            resp = await get_client().task_service.ListTasks(
                task_service_pb2.ListTasksRequest(
                    org=cfg.org,
                    project_id=identifier_pb2.ProjectIdentifier(
                        organization=cfg.org,
                        domain=domain or cfg.domain,
                        name=project or cfg.project,
                    ),
                    request=list_pb2.ListRequest(
                        sort_by=sort_pb2,
                        filters=filters,
                        limit=limit,
                        token=token,
                    ),
                )
            )
            token = resp.token
            for t in resp.tasks:
                retrieved += 1
                yield cls(t)
            if not token or retrieved >= original_limit:
                logger.debug(f"Retrieved {retrieved} tasks, stopping iteration.")
                break

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the task.
        """
        yield "project", self.pb2.task_id.project
        yield "domain", self.pb2.task_id.domain
        yield "name", self.pb2.task_id.name
        yield "version", self.pb2.task_id.version
        yield "short_name", self.pb2.metadata.short_name
        for t in _repr_task_metadata(self.pb2.metadata):
            yield t


if __name__ == "__main__":
    tk = Task.get(name="example_task")
