import inspect
import os
import sys
from http import HTTPStatus
from typing import Callable, Dict, List, Tuple, Type, Union

import grpc
from flyteidl2.core.security_pb2 import Connection
from flyteidl2.plugins.connector_pb2 import (
    CreateTaskRequest,
    CreateTaskResponse,
    DeleteTaskRequest,
    DeleteTaskResponse,
    GetConnectorRequest,
    GetConnectorResponse,
    GetTaskLogsRequest,
    GetTaskLogsResponse,
    GetTaskMetricsRequest,
    GetTaskMetricsResponse,
    GetTaskRequest,
    GetTaskResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
)
from flyteidl2.service.connector_pb2_grpc import (
    AsyncConnectorServiceServicer,
    ConnectorMetadataServiceServicer,
)
from prometheus_client import Counter, Summary

from flyte._internal.runtime.convert import Inputs, convert_from_inputs_to_native
from flyte._logging import logger
from flyte.connectors._connector import ConnectorRegistry, FlyteConnectorNotFound, get_resource_proto
from flyte.connectors.utils import _start_grpc_server
from flyte.models import NativeInterface, _has_default
from flyte.syncify import syncify
from flyte.types import TypeEngine

metric_prefix = "flyte_connector_"
create_operation = "create"
get_operation = "get"
delete_operation = "delete"

# Follow the naming convention. https://prometheus.io/docs/practices/naming/
request_success_count = Counter(
    f"{metric_prefix}requests_success_total",
    "Total number of successful requests",
    ["task_type", "operation"],
)
request_failure_count = Counter(
    f"{metric_prefix}requests_failure_total",
    "Total number of failed requests",
    ["task_type", "operation", "error_code"],
)
request_latency = Summary(
    f"{metric_prefix}request_latency_seconds",
    "Time spent processing connector request",
    ["task_type", "operation"],
)
input_literal_size = Summary(f"{metric_prefix}input_literal_bytes", "Size of input literal", ["task_type"])


def _handle_exception(e: Exception, context: grpc.ServicerContext, task_type: str, operation: str):
    if isinstance(e, FlyteConnectorNotFound):
        error_message = f"Cannot find connector for task type: {task_type}."
        logger.error(error_message)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(error_message)
        request_failure_count.labels(task_type=task_type, operation=operation, error_code=HTTPStatus.NOT_FOUND).inc()
    else:
        error_message = f"failed to {operation} {task_type} task with error:\n {e}."
        logger.error(error_message)
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(error_message)
        request_failure_count.labels(
            task_type=task_type, operation=operation, error_code=HTTPStatus.INTERNAL_SERVER_ERROR
        ).inc()


class ConnectorService:
    @syncify
    @classmethod
    async def run(cls, port: int, prometheus_port: int, worker: int, timeout: int | None, modules: List[str] | None):
        working_dir = os.getcwd()
        if all(os.path.realpath(path) != working_dir for path in sys.path):
            sys.path.append(working_dir)
        await _start_grpc_server(port, prometheus_port, worker, timeout, modules)


def record_connector_metrics(func: Callable):
    async def wrapper(
        self,
        request: Union[CreateTaskRequest, GetTaskRequest, DeleteTaskRequest],
        context: grpc.ServicerContext,
        *args,
        **kwargs,
    ):
        if isinstance(request, CreateTaskRequest):
            task_type = request.template.type
            operation = create_operation
            if request.inputs:
                input_literal_size.labels(task_type=task_type).observe(request.inputs.ByteSize())
        elif isinstance(request, GetTaskRequest):
            task_type = request.task_category.name
            operation = get_operation
        elif isinstance(request, DeleteTaskRequest):
            task_type = request.task_category.name
            operation = delete_operation
        else:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Method not implemented!")
            return None

        try:
            with request_latency.labels(task_type=task_type, operation=operation).time():
                res = await func(self, request, context, *args, **kwargs)
            request_success_count.labels(task_type=task_type, operation=operation).inc()
            return res
        except Exception as e:
            _handle_exception(e, context, task_type, operation)

    return wrapper


def _get_connection_kwargs(request: Connection) -> Dict[str, str]:
    kwargs = {}

    for k, v in request.secrets.items():
        kwargs[k] = v
    for k, v in request.configs.items():
        kwargs[k] = v

    return kwargs


class AsyncConnectorService(AsyncConnectorServiceServicer):
    @record_connector_metrics
    async def CreateTask(self, request: CreateTaskRequest, context: grpc.ServicerContext) -> CreateTaskResponse:
        template = request.template
        connector = ConnectorRegistry.get_connector(template.type, template.task_type_version)
        logger.info(f"{connector.name} start creating the job")
        python_interface_inputs: Dict[str, Tuple[Type, Type[_has_default] | Type[inspect._empty]]] = {
            name: (TypeEngine.guess_python_type(lt.type), inspect.Parameter.empty)
            for name, lt in template.interface.inputs.variables.items()
        }
        native_interface = NativeInterface.from_types(inputs=python_interface_inputs, outputs={})
        native_inputs = await convert_from_inputs_to_native(native_interface, Inputs(proto_inputs=request.inputs))
        resource_meta = await connector.create(
            task_template=request.template,
            inputs=native_inputs,
            output_prefix=request.output_prefix,
            task_execution_metadata=request.task_execution_metadata,
            connection=_get_connection_kwargs(request.connection),
        )
        return CreateTaskResponse(resource_meta=resource_meta.encode())

    @record_connector_metrics
    async def GetTask(self, request: GetTaskRequest, context: grpc.ServicerContext) -> GetTaskResponse:
        connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
        logger.info(f"{connector.name} start checking the status of the job")
        res = await connector.get(
            resource_meta=connector.metadata_type.decode(request.resource_meta),
            connection=_get_connection_kwargs(request.connection),
        )
        return GetTaskResponse(resource=await get_resource_proto(res))

    @record_connector_metrics
    async def DeleteTask(self, request: DeleteTaskRequest, context: grpc.ServicerContext) -> DeleteTaskResponse:
        connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
        logger.info(f"{connector.name} start deleting the job")
        await connector.delete(
            resource_meta=connector.metadata_type.decode(request.resource_meta),
            connection=_get_connection_kwargs(request.connection),
        )
        return DeleteTaskResponse()

    async def GetTaskMetrics(
        self, request: GetTaskMetricsRequest, context: grpc.ServicerContext
    ) -> GetTaskMetricsResponse:
        connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
        logger.info(f"{connector.name} start getting metrics of the job")
        return await connector.get_metrics(resource_meta=connector.metadata_type.decode(request.resource_meta))

    async def GetTaskLogs(self, request: GetTaskLogsRequest, context: grpc.ServicerContext) -> GetTaskLogsResponse:
        connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
        logger.info(f"{connector.name} start getting logs of the job")
        return await connector.get_logs(resource_meta=connector.metadata_type.decode(request.resource_meta))


class ConnectorMetadataService(ConnectorMetadataServiceServicer):
    async def GetConnector(self, request: GetConnectorRequest, context: grpc.ServicerContext) -> GetConnectorResponse:
        return GetConnectorResponse(connector=ConnectorRegistry._get_connector_metadata(request.name))

    async def ListConnectors(
        self, request: ListConnectorsRequest, context: grpc.ServicerContext
    ) -> ListConnectorsResponse:
        return ListConnectorsResponse(connectors=ConnectorRegistry._list_connectors())
