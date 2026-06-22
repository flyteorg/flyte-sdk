import inspect
import os
import sys
from http import HTTPStatus
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from flyteidl2.connector.connector_pb2 import (
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
from flyteidl2.connector.service_pb2_grpc import AsyncConnectorServiceServicer, ConnectorMetadataServiceServicer
from flyteidl2.core.security_pb2 import Connection
from prometheus_client import Counter, Summary

from flyte._internal.runtime.convert import Inputs, convert_from_inputs_to_native
from flyte._logging import logger
from flyte.connectors._connector import ConnectorRegistry, FlyteConnectorNotFound, get_resource_proto
from flyte.connectors._grpc import grpc
from flyte.connectors.utils import _start_connector_servers
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


def _get_connection_kwargs(request: Connection) -> Dict[str, str]:
    kwargs = {}

    for k, v in request.secrets.items():
        kwargs[k] = v
    for k, v in request.configs.items():
        kwargs[k] = v

    return kwargs


def _extract_metrics_meta(
    request: Union[CreateTaskRequest, GetTaskRequest, DeleteTaskRequest],
) -> Tuple[Optional[str], str, Optional[int]]:
    """Resolve ``(task_type, operation, input_literal_bytes)`` for metric labels.

    ``task_type`` is ``None`` for unrecognized request types so the caller can
    surface an "unimplemented" error (in which case ``operation`` is unused).
    This is shared by the gRPC and Connect metric wrappers so both transports
    label metrics identically.
    """
    if isinstance(request, CreateTaskRequest):
        input_size = request.inputs.ByteSize() if request.inputs else None
        return request.template.type, create_operation, input_size
    if isinstance(request, GetTaskRequest):
        return request.task_category.name, get_operation, None
    if isinstance(request, DeleteTaskRequest):
        return request.task_category.name, delete_operation, None
    return None, "", None


# ---------------------------------------------------------------------------
# Transport-agnostic business logic.
#
# These handlers contain the actual connector logic and raise plain exceptions
# (e.g. ``FlyteConnectorNotFound``). The gRPC servicers below and the Connect
# services in ``_connect_server`` are thin adapters that wrap these with their
# transport-specific metrics and error mapping, so the two servers never drift.
# ---------------------------------------------------------------------------
async def _do_create_task(request: CreateTaskRequest) -> CreateTaskResponse:
    template = request.template
    connector = ConnectorRegistry.get_connector(template.type, template.task_type_version)
    logger.info(f"{connector.name} start creating the job")
    python_interface_inputs: Dict[str, Tuple[Type, Type[_has_default] | Type[inspect._empty]]] = {
        entry.key: (TypeEngine.guess_python_type(entry.value.type), inspect.Parameter.empty)
        for entry in template.interface.inputs.variables
    }
    native_interface = NativeInterface.from_types(inputs=python_interface_inputs, outputs={})
    native_inputs = await convert_from_inputs_to_native(native_interface, Inputs(proto_inputs=request.inputs))
    resource_meta = await connector.create(
        task_template=request.template,
        inputs=native_inputs,
        output_prefix=request.output_prefix,
        task_execution_metadata=request.task_execution_metadata,
        **_get_connection_kwargs(request.connection),
    )
    return CreateTaskResponse(resource_meta=resource_meta.encode())


async def _do_get_task(request: GetTaskRequest) -> GetTaskResponse:
    connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
    logger.info(f"{connector.name} start checking the status of the job")
    res = await connector.get(
        resource_meta=connector.metadata_type.decode(request.resource_meta),
        **_get_connection_kwargs(request.connection),
    )
    return GetTaskResponse(resource=await get_resource_proto(res))


async def _do_delete_task(request: DeleteTaskRequest) -> DeleteTaskResponse:
    connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
    logger.info(f"{connector.name} start deleting the job")
    await connector.delete(
        resource_meta=connector.metadata_type.decode(request.resource_meta),
        **_get_connection_kwargs(request.connection),
    )
    return DeleteTaskResponse()


async def _do_get_task_metrics(request: GetTaskMetricsRequest) -> GetTaskMetricsResponse:
    connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
    logger.info(f"{connector.name} start getting metrics of the job")
    return await connector.get_metrics(resource_meta=connector.metadata_type.decode(request.resource_meta))


async def _do_get_task_logs(request: GetTaskLogsRequest):
    connector = ConnectorRegistry.get_connector(request.task_category.name, request.task_category.version)
    logger.info(f"{connector.name} start getting logs of the job")
    # `get_logs` may be either:
    #   - an async generator yielding multiple GetTaskLogsResponse messages
    #     (preferred — supports interleaved body/header/body pagination, since
    #     proto3 oneof keeps only one of body/header per message),
    #   - or an async function returning a single GetTaskLogsResponse.
    result = connector.get_logs(
        resource_meta=connector.metadata_type.decode(request.resource_meta),
        token=request.token,
    )
    if inspect.isasyncgen(result):
        async for msg in result:
            yield msg
        return
    response = await result
    yield response


def _do_get_connector(request: GetConnectorRequest) -> GetConnectorResponse:
    return GetConnectorResponse(connector=ConnectorRegistry._get_connector_metadata(request.name))


def _do_list_connectors(request: ListConnectorsRequest) -> ListConnectorsResponse:
    return ListConnectorsResponse(connectors=ConnectorRegistry._list_connectors())


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
    async def run(
        cls,
        port: int,
        connect_port: int,
        prometheus_port: int,
        worker: int,
        timeout: int | None,
        modules: List[str] | None,
    ):
        working_dir = os.getcwd()
        if all(os.path.realpath(path) != working_dir for path in sys.path):
            sys.path.append(working_dir)
        await _start_connector_servers(port, connect_port, prometheus_port, worker, timeout, modules)


def record_connector_metrics(func: Callable):
    async def wrapper(
        self,
        request: Union[CreateTaskRequest, GetTaskRequest, DeleteTaskRequest],
        context: grpc.ServicerContext,
        *args,
        **kwargs,
    ):
        task_type, operation, input_size = _extract_metrics_meta(request)
        if task_type is None:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Method not implemented!")
            return None
        if input_size is not None:
            input_literal_size.labels(task_type=task_type).observe(input_size)

        try:
            with request_latency.labels(task_type=task_type, operation=operation).time():
                res = await func(self, request, context, *args, **kwargs)
            request_success_count.labels(task_type=task_type, operation=operation).inc()
            return res
        except Exception as e:
            _handle_exception(e, context, task_type, operation)

    return wrapper


class AsyncConnectorService(AsyncConnectorServiceServicer):
    @record_connector_metrics
    async def CreateTask(self, request: CreateTaskRequest, context: grpc.ServicerContext) -> CreateTaskResponse:
        return await _do_create_task(request)

    @record_connector_metrics
    async def GetTask(self, request: GetTaskRequest, context: grpc.ServicerContext) -> GetTaskResponse:
        return await _do_get_task(request)

    @record_connector_metrics
    async def DeleteTask(self, request: DeleteTaskRequest, context: grpc.ServicerContext) -> DeleteTaskResponse:
        return await _do_delete_task(request)

    async def GetTaskMetrics(
        self, request: GetTaskMetricsRequest, context: grpc.ServicerContext
    ) -> GetTaskMetricsResponse:
        return await _do_get_task_metrics(request)

    async def GetTaskLogs(self, request: GetTaskLogsRequest, context: grpc.ServicerContext) -> GetTaskLogsResponse:
        async for msg in _do_get_task_logs(request):
            yield msg


class ConnectorMetadataService(ConnectorMetadataServiceServicer):
    async def GetConnector(self, request: GetConnectorRequest, context: grpc.ServicerContext) -> GetConnectorResponse:
        return _do_get_connector(request)

    async def ListConnectors(
        self, request: ListConnectorsRequest, context: grpc.ServicerContext
    ) -> ListConnectorsResponse:
        return _do_list_connectors(request)
