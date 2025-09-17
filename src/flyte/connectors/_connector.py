import asyncio
import inspect
import json
import sys
import time
import typing
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import partial
from types import FrameType
from typing import Any, Dict, List, Optional, Union

from flyteidl.admin.agent_pb2 import Agent, GetTaskLogsResponse, GetTaskMetricsResponse, TaskExecutionMetadata
from flyteidl.admin.agent_pb2 import Resource as _Resource
from flyteidl.admin.agent_pb2 import TaskCategory as _TaskCategory
from flyteidl.core import literals_pb2
from flyteidl.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl.core.literals_pb2 import LiteralMap
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from rich.logging import RichHandler
from rich.progress import Progress
from flyte._logging import logger

from flyte._task import TaskTemplate
from flyte.types._type_engine import dataclass_from_dict, TypeEngine

# It's used to force connector to run in the same event loop in the local execution.
local_connector_loop = asyncio.new_event_loop()


@dataclass(frozen=True)
class TaskCategory:
    name: str
    version: int = 0


@dataclass
class ResourceMeta:
    """
    This is the metadata for the job. For example, the id of the job.
    """

    def encode(self) -> bytes:
        """
        Encode the resource meta to bytes.
        """
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> "ResourceMeta":
        """
        Decode the resource meta from bytes.
        """
        return dataclass_from_dict(cls, json.loads(data.decode("utf-8")))


@dataclass
class Resource:
    """
    This is the output resource of the job.

    Attributes
    ----------
        phase : TaskExecution.Phase
            The phase of the job.
        message : Optional[str]
            The return message from the job.
        log_links : Optional[List[TaskLog]]
            The log links of the job. For example, the link to the BigQuery Console.
        outputs : Optional[Union[LiteralMap, typing.Dict[str, Any]]]
            The outputs of the job. If return python native types, the agent will convert them to flyte literals.
        custom_info : Optional[typing.Dict[str, Any]]
            The custom info of the job. For example, the job config.
    """

    phase: TaskExecution.Phase
    message: Optional[str] = None
    log_links: Optional[List[TaskLog]] = None
    outputs: Optional[Union[LiteralMap, typing.Dict[str, Any]]] = None
    custom_info: Optional[typing.Dict[str, Any]] = None


class AsyncConnector(ABC):
    """
    This is the base class for all async connectors.
    It defines the interface that all connectors must implement.
    The connector service is responsible for invoking connectors. The propeller will communicate with the connector service
    to create tasks, get the status of tasks, and delete tasks.

    All the connectors should be registered in the ConnectorRegistry.
    Connector Service
    will look up the connector based on the task type. Every task type can only have one connector.
    """

    name = "Async Connector"
    task_type_name: str
    task_type_version: int = 0
    metadata_type: ResourceMeta

    @property
    def task_category(self) -> TaskCategory:
        return TaskCategory(name=self.task_type_name, version=self.task_type_version)

    @abstractmethod
    def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[LiteralMap],
        task_execution_metadata: Optional[TaskExecutionMetadata],
        **kwargs,
    ) -> ResourceMeta:
        """
        Return a resource meta that can be used to get the status of the task.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, resource_meta: ResourceMeta, **kwargs) -> Resource:
        """
        Return the status of the task, and return the outputs in some cases. For example, bigquery job
        can't write the structured dataset to the output location, so it returns the output literals to the propeller,
        and the propeller will write the structured dataset to the blob store.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, resource_meta: ResourceMeta, **kwargs):
        """
        Delete the task. This call should be idempotent. It should raise an error if fails to delete the task.
        """
        raise NotImplementedError

    def get_metrics(self, resource_meta: ResourceMeta, **kwargs) -> GetTaskMetricsResponse:
        """
        Return the metrics for the task.
        """
        raise NotImplementedError

    def get_logs(self, resource_meta: ResourceMeta, **kwargs) -> GetTaskLogsResponse:
        """
        Return the metrics for the task.
        """
        raise NotImplementedError


class ConnectorRegistry(object):
    """
    This is the registry for all connectors.
    The connector service will look up the connector registry based on the task type.
    The connector metadata service will look up the connector metadata based on the connector name.
    """

    _REGISTRY: Dict[TaskCategory, AsyncConnector] = {}
    _METADATA: Dict[str, Agent] = {}

    @staticmethod
    def register(connector: AsyncConnector, override: bool = False):
        if connector.task_category in ConnectorRegistry._REGISTRY and override is False:
            raise ValueError(f"Duplicate connector for task type: {connector.task_category}")
        ConnectorRegistry._REGISTRY[connector.task_category] = connector

        task_category = _TaskCategory(name=connector.task_category.name, version=connector.task_category.version)

        if connector.name in ConnectorRegistry._METADATA:
            connector_metadata = ConnectorRegistry.get_connector_metadata(connector.name)
            connector_metadata.supported_task_categories.append(task_category)
            connector_metadata.supported_task_types.append(task_category.name)
        else:
            connector_metadata = Agent(
                name=connector.name,
                supported_task_types=[task_category.name],
                supported_task_categories=[task_category],
            )
            ConnectorRegistry._METADATA[connector.name] = connector_metadata

    @staticmethod
    def get_connector(task_type_name: str, task_type_version: int = 0) -> AsyncConnector:
        task_category = TaskCategory(name=task_type_name, version=task_type_version)
        if task_category not in ConnectorRegistry._REGISTRY:
            raise ValueError(f"Cannot find connector for task category: {task_category}.")
        return ConnectorRegistry._REGISTRY[task_category]

    @staticmethod
    def list_connectors() -> List[Agent]:
        return list(ConnectorRegistry._METADATA.values())

    @staticmethod
    def get_connector_metadata(name: str) -> Agent:
        if name not in ConnectorRegistry._METADATA:
            raise ValueError(f"Cannot find connector for name: {name}.")
        return ConnectorRegistry._METADATA[name]


class AsyncConnectorExecutorMixin:
    """
    This mixin class is used to run the async task locally, and it's only used for local execution.
    Task should inherit from this class if the task can be run in the connector.

    Asynchronous tasks are tasks that take a long time to complete, such as running a query.
    """

    T = typing.TypeVar("T", PythonTask, "AsyncConnectorExecutorMixin")

    _clean_up_task: bool = False
    _connector: AsyncConnectorBase = None
    resource_meta = None

    def execute(self: T, **kwargs) -> LiteralMap:
        ctx = FlyteContext.current_context()
        ss = ctx.serialization_settings or SerializationSettings(ImageConfig())
        output_prefix = ctx.file_access.get_random_remote_directory()
        from flytekit.tools.translator import get_serializable

        task_template = get_serializable(OrderedDict(), ss, self).template
        if task_template.metadata.timeout:
            logger.info("Timeout is not supported for local execution.\n" "Ignoring the timeout.")
        self._connector = ConnectorRegistry.get_connector(task_template.type, task_template.task_type_version)

        resource_meta = local_connector_loop.run_until_complete(
            self._create(task_template=task_template, output_prefix=output_prefix, inputs=kwargs)
        )
        resource = local_connector_loop.run_until_complete(self._get(resource_meta=resource_meta))

        if resource.phase != TaskExecution.SUCCEEDED:
            raise RuntimeError(f"Failed to run the task {self.name} with error: {resource.message}")

        # Read the literals from a remote file if the connector doesn't return the output literals.
        if task_template.interface.outputs and resource.outputs is None:
            local_outputs_file = ctx.file_access.get_random_local_path()
            ctx.file_access.get_data(f"{output_prefix}/outputs.pb", local_outputs_file)
            output_proto = utils.load_proto_from_file(literals_pb2.LiteralMap, local_outputs_file)
            return LiteralMap.from_flyte_idl(output_proto)

        if resource.outputs and not isinstance(resource.outputs, LiteralMap):
            return TypeEngine.dict_to_literal_map(ctx, resource.outputs)  # type: ignore

        # TODO: return
        return resource.outputs

    async def _create(
        self: T, task_template: TaskTemplate, output_prefix: str, inputs: Dict[str, Any] = None
    ) -> ResourceMeta:
        ctx = FlyteContext.current_context()
        if isinstance(self, PythonFunctionTask):
            es = ctx.new_execution_state().with_params(mode=ExecutionState.Mode.TASK_EXECUTION)
            cb = ctx.new_builder().with_execution_state(es)

            with FlyteContextManager.with_context(cb) as ctx:
                # Write the inputs to a remote file, so that the remote task can read the inputs from this file.
                literal_map = await TypeEngine._dict_to_literal_map(ctx, inputs or {}, self.get_input_types())
                path = ctx.file_access.get_random_local_path()
                utils.write_proto_to_file(literal_map.to_flyte_idl(), path)
                await ctx.file_access.async_put_data(path, f"{output_prefix}/inputs.pb")
                task_template = render_task_template(task_template, output_prefix)
        else:
            literal_map = TypeEngine.dict_to_literal_map(ctx, inputs or {}, self.get_input_types())

        resource_meta = await mirror_async_methods(
            self._connector.create,
            task_template=task_template,
            inputs=literal_map,
            output_prefix=output_prefix,
        )

        FlyteContextManager.add_signal_handler(partial(self.connector_signal_handler, resource_meta))
        self.resource_meta = resource_meta
        return resource_meta

    async def _get(self: T, resource_meta: ResourceMeta) -> Resource:
        phase = TaskExecution.RUNNING

        progress = Progress(transient=True)
        set_flytekit_log_properties(RichHandler(log_time_format="%H:%M:%S.%f"), None, None)
        task = progress.add_task(f"[cyan]Running Task {self.name}...", total=None)
        task_phase = progress.add_task("[cyan]Task phase: RUNNING, Phase message: ", total=None, visible=False)
        task_log_links = progress.add_task("[cyan]Log Links: ", total=None, visible=False)
        with progress:
            while not is_terminal_phase(phase):
                progress.start_task(task)
                time.sleep(1)
                resource = await mirror_async_methods(self._connector.get, resource_meta=resource_meta)
                if self._clean_up_task:
                    sys.exit(1)

                phase = resource.phase
                progress.update(
                    task_phase,
                    description=f"[cyan]Task phase: {TaskExecution.Phase.Name(phase)}, Phase message: {resource.message}",
                    visible=True,
                )
                if resource.log_links:
                    log_links = ""
                    for link in resource.log_links:
                        log_links += f"{link.name}: {link.uri}\n"
                    if log_links:
                        progress.update(task_log_links, description=f"[cyan]{log_links}", visible=True)

        return resource

    def connector_signal_handler(self, resource_meta: ResourceMeta, signum: int, frame: FrameType) -> Any:
        if inspect.iscoroutinefunction(self._connector.delete):
            # Use asyncio.run to run the async function in the main thread since the loop manager is killed when the
            # signal is received.
            asyncio.run(self._connector.delete(resource_meta=resource_meta))
        else:
            self._connector.delete(resource_meta)
        self._clean_up_task = True
