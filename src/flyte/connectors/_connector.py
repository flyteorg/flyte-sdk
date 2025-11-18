import asyncio
import json
import typing
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from flyteidl2.core import tasks_pb2
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.plugins import connector_pb2
from flyteidl2.plugins.connector_pb2 import Connector as ConnectorProto
from flyteidl2.plugins.connector_pb2 import (
    GetTaskLogsResponse,
    GetTaskMetricsResponse,
    TaskCategory,
    TaskExecutionMetadata,
)
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from flyte import Secret
from flyte._context import internal_ctx
from flyte._initialize import get_init_config
from flyte._internal.runtime.convert import convert_from_native_to_outputs
from flyte._internal.runtime.task_serde import get_proto_task
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte.connectors.utils import is_terminal_phase
from flyte.models import NativeInterface, SerializationContext
from flyte.types._type_engine import dataclass_from_dict


@dataclass(frozen=True)
class ConnectorRegistryKey:
    task_type_name: str
    task_type_version: int


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
    outputs: Optional[Dict[str, Any]] = None
    custom_info: Optional[typing.Dict[str, Any]] = None


class AsyncConnector(ABC):
    """
    This is the base class for all async connectors, and it defines the interface that all connectors must implement.
    The connector service is responsible for invoking connectors.
    The executor will communicate with the connector service to create tasks, get the status of tasks, and delete tasks.

    All the connectors should be registered in the ConnectorRegistry.
    Connector Service will look up the connector based on the task type and version.
    """

    name = "Async Connector"
    task_type_name: str
    task_type_version: int = 0
    metadata_type: ResourceMeta

    @abstractmethod
    async def create(
        self,
        task_template: tasks_pb2.TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, typing.Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs,
    ) -> ResourceMeta:
        """
        Return a resource meta that can be used to get the status of the task.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, resource_meta: ResourceMeta, **kwargs) -> Resource:
        """
        Return the status of the task, and return the outputs in some cases. For example, bigquery job
        can't write the structured dataset to the output location, so it returns the output literals to the propeller,
        and the propeller will write the structured dataset to the blob store.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, resource_meta: ResourceMeta, **kwargs):
        """
        Delete the task. This call should be idempotent. It should raise an error if fails to delete the task.
        """
        raise NotImplementedError

    async def get_metrics(self, resource_meta: ResourceMeta, **kwargs) -> GetTaskMetricsResponse:
        """
        Return the metrics for the task.
        """
        raise NotImplementedError

    async def get_logs(self, resource_meta: ResourceMeta, **kwargs) -> GetTaskLogsResponse:
        """
        Return the metrics for the task.
        """
        raise NotImplementedError


class ConnectorRegistry(object):
    """
    This is the registry for all connectors.
    The connector service will look up the connector registry based on the task type and version.
    """

    _REGISTRY: typing.ClassVar[Dict[ConnectorRegistryKey, AsyncConnector]] = {}
    _METADATA: typing.ClassVar[Dict[str, ConnectorProto]] = {}

    @staticmethod
    def register(connector: AsyncConnector, override: bool = False):
        key = ConnectorRegistryKey(
            task_type_name=connector.task_type_name, task_type_version=connector.task_type_version
        )
        if key in ConnectorRegistry._REGISTRY and override is False:
            raise ValueError(
                f"Duplicate connector for task type: {connector.task_type_name}"
                f" and version: {connector.task_type_version}"
            )
        ConnectorRegistry._REGISTRY[key] = connector

        task_category = TaskCategory(name=connector.task_type_name, version=connector.task_type_version)

        if connector.name in ConnectorRegistry._METADATA:
            connector_metadata = ConnectorRegistry._get_connector_metadata(connector.name)
            connector_metadata.supported_task_categories.append(task_category)
        else:
            connector_metadata = ConnectorProto(
                name=connector.name,
                supported_task_categories=[task_category],
            )
            ConnectorRegistry._METADATA[connector.name] = connector_metadata

    @staticmethod
    def get_connector(task_type_name: str, task_type_version: int = 0) -> AsyncConnector:
        key = ConnectorRegistryKey(task_type_name=task_type_name, task_type_version=task_type_version)
        if key not in ConnectorRegistry._REGISTRY:
            raise FlyteConnectorNotFound(
                f"Cannot find connector for task type: {task_type_name} and version: {task_type_version}"
            )
        return ConnectorRegistry._REGISTRY[key]

    @staticmethod
    def _list_connectors() -> List[ConnectorProto]:
        return list(ConnectorRegistry._METADATA.values())

    @staticmethod
    def _get_connector_metadata(name: str) -> ConnectorProto:
        if name not in ConnectorRegistry._METADATA:
            raise FlyteConnectorNotFound(f"Cannot find connector for name: {name}.")
        return ConnectorRegistry._METADATA[name]


class ConnectorSecretsMixin:
    def __init__(self, secrets: Dict[str, str]):
        # Key is the id of the secret, value is the secret name.
        self._secrets = secrets

    @property
    def secrets(self) -> List[Secret]:
        return [Secret(key=k, as_env_var=v) for k, v in self._secrets.items()]


class AsyncConnectorExecutorMixin:
    """
    This mixin class is used to run the connector task locally, and it's only used for local execution.
    Task should inherit from this class if the task can be run in the connector.
    """

    async def execute(self, **kwargs) -> Any:
        task = typing.cast(TaskTemplate, self)
        connector = ConnectorRegistry.get_connector(task.task_type, task.task_type_version)

        ctx = internal_ctx()
        tctx = internal_ctx().data.task_context
        cfg = get_init_config()

        if tctx is None:
            raise RuntimeError("Task context is not set.")

        sc = SerializationContext(
            project=tctx.action.project,
            domain=tctx.action.domain,
            org=tctx.action.org,
            code_bundle=tctx.code_bundle,
            version=tctx.version,
            image_cache=tctx.compiled_image_cache,
            root_dir=cfg.root_dir,
        )
        tt = get_proto_task(task, sc)
        resource_meta = await connector.create(task_template=tt, output_prefix=ctx.raw_data.path, inputs=kwargs)
        resource = Resource(phase=TaskExecution.RUNNING)

        while not is_terminal_phase(resource.phase):
            resource = await connector.get(resource_meta=resource_meta)

            if resource.log_links:
                for link in resource.log_links:
                    logger.info(f"{link.name}: {link.uri}")
            await asyncio.sleep(1)

        if resource.phase != TaskExecution.SUCCEEDED:
            raise RuntimeError(f"Failed to run the task {task.name} with error: {resource.message}")

        # TODO: Support abort

        if resource.outputs is None:
            return None
        return tuple(resource.outputs.values())


async def get_resource_proto(resource: Resource) -> connector_pb2.Resource:
    if resource.outputs:
        interface = NativeInterface.from_types(inputs={}, outputs={k: type(v) for k, v in resource.outputs.items()})
        outputs = await convert_from_native_to_outputs(tuple(resource.outputs.values()), interface)
    else:
        outputs = None

    return connector_pb2.Resource(
        phase=resource.phase,
        message=resource.message,
        log_links=resource.log_links,
        outputs=outputs,
        custom_info=(json_format.Parse(json.dumps(resource.custom_info), Struct()) if resource.custom_info else None),
    )


class FlyteConnectorNotFound(ValueError): ...
