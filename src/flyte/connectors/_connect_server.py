"""Connect (HTTP/1.1) transport for the Flyte connector service.

This serves the same connector RPCs as the gRPC server in ``_server.py`` but
over the Connect protocol via ASGI, so it can run behind a plain HTTP server
(uvicorn). The two servers run side by side during the migration away from gRPC
(see ``utils._start_connector_servers``); both delegate to the shared ``_do_*``
business-logic handlers in ``_server.py`` so they never drift.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Callable, Iterable, Tuple

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from flyteidl2.connector.connector_pb2 import (
    CreateTaskRequest,
    CreateTaskResponse,
    DeleteTaskRequest,
    DeleteTaskResponse,
    GetConnectorRequest,
    GetConnectorResponse,
    GetTaskLogsRequest,
    GetTaskMetricsRequest,
    GetTaskMetricsResponse,
    GetTaskRequest,
    GetTaskResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
)
from flyteidl2.connector.service_connect import (
    AsyncConnectorServiceASGIApplication,
    ConnectorMetadataServiceASGIApplication,
)

from flyte._logging import logger
from flyte.connectors._connector import FlyteConnectorNotFound
from flyte.connectors._server import (
    _do_create_task,
    _do_delete_task,
    _do_get_connector,
    _do_get_task,
    _do_get_task_logs,
    _do_get_task_metrics,
    _do_list_connectors,
    _extract_metrics_meta,
    input_literal_size,
    request_failure_count,
    request_latency,
    request_success_count,
)

HEALTH_CHECK_PATHS = ("/healthz", "/health")


def _to_connect_error(e: Exception, task_type: str, operation: str) -> ConnectError:
    """Map a handler exception to a ``ConnectError`` and record the failure metric.

    This is the Connect counterpart of ``_server._handle_exception`` (which
    mutates a gRPC ``ServicerContext``); both increment the same Prometheus
    failure counter with matching labels.
    """
    if isinstance(e, FlyteConnectorNotFound):
        error_message = f"Cannot find connector for task type: {task_type}."
        logger.error(error_message)
        request_failure_count.labels(task_type=task_type, operation=operation, error_code=HTTPStatus.NOT_FOUND).inc()
        return ConnectError(Code.NOT_FOUND, error_message)
    error_message = f"failed to {operation} {task_type} task with error:\n {e}."
    logger.error(error_message)
    request_failure_count.labels(
        task_type=task_type, operation=operation, error_code=HTTPStatus.INTERNAL_SERVER_ERROR
    ).inc()
    return ConnectError(Code.INTERNAL, error_message)


def _record_connect_metrics(func: Callable):
    """Connect counterpart of ``_server.record_connector_metrics``.

    Records the same Prometheus metrics, but surfaces failures as
    ``ConnectError`` instead of mutating a gRPC ``ServicerContext``.
    """

    async def wrapper(self, request, ctx: RequestContext, *args, **kwargs):
        task_type, operation, input_size = _extract_metrics_meta(request)
        if task_type is None:
            raise ConnectError(Code.UNIMPLEMENTED, "Method not implemented!")
        if input_size is not None:
            input_literal_size.labels(task_type=task_type).observe(input_size)

        try:
            with request_latency.labels(task_type=task_type, operation=operation).time():
                res = await func(self, request, ctx, *args, **kwargs)
            request_success_count.labels(task_type=task_type, operation=operation).inc()
            return res
        except Exception as e:
            raise _to_connect_error(e, task_type, operation) from e

    return wrapper


class ConnectAsyncConnectorService:
    """Connect implementation of ``flyteidl2.connector.AsyncConnectorService``."""

    @_record_connect_metrics
    async def create_task(self, request: CreateTaskRequest, ctx: RequestContext) -> CreateTaskResponse:
        return await _do_create_task(request)

    @_record_connect_metrics
    async def get_task(self, request: GetTaskRequest, ctx: RequestContext) -> GetTaskResponse:
        return await _do_get_task(request)

    @_record_connect_metrics
    async def delete_task(self, request: DeleteTaskRequest, ctx: RequestContext) -> DeleteTaskResponse:
        return await _do_delete_task(request)

    async def get_task_metrics(self, request: GetTaskMetricsRequest, ctx: RequestContext) -> GetTaskMetricsResponse:
        return await _do_get_task_metrics(request)

    async def get_task_logs(self, request: GetTaskLogsRequest, ctx: RequestContext):
        # Server-streaming: connectRPC calls this (without awaiting) and iterates
        # the returned async iterator, so it must be an async generator.
        async for msg in _do_get_task_logs(request):
            yield msg


class ConnectConnectorMetadataService:
    """Connect implementation of ``flyteidl2.connector.ConnectorMetadataService``."""

    async def get_connector(self, request: GetConnectorRequest, ctx: RequestContext) -> GetConnectorResponse:
        try:
            return _do_get_connector(request)
        except FlyteConnectorNotFound as e:
            raise ConnectError(Code.NOT_FOUND, str(e)) from e

    async def list_connectors(self, request: ListConnectorsRequest, ctx: RequestContext) -> ListConnectorsResponse:
        return _do_list_connectors(request)


async def _respond(send, status: HTTPStatus, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": int(status),
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class _ConnectorASGIApp:
    """Minimal ASGI router for the connector.

    Dispatches Connect RPC paths to the generated service applications, answers
    health-check probes with ``200``, and ``404``s everything else. Handles the
    ``lifespan`` scope directly because both services are plain instances whose
    endpoints resolve lazily on the first request.
    """

    def __init__(
        self,
        applications: Iterable[Tuple[str, object]],
        health_paths: Iterable[str] = HEALTH_CHECK_PATHS,
    ):
        # applications: iterable of (path_prefix, asgi_app)
        self._applications = tuple(applications)
        self._health_paths = tuple(health_paths)

    async def __call__(self, scope, receive, send):
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope_type != "http":
            return

        path = scope.get("path", "")
        if path in self._health_paths:
            await _respond(send, HTTPStatus.OK, b"OK")
            return

        for prefix, app in self._applications:
            if path == prefix or path.startswith(prefix + "/"):
                await app(scope, receive, send)
                return

        await _respond(send, HTTPStatus.NOT_FOUND, b"Not Found")

    @staticmethod
    async def _handle_lifespan(receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


def build_asgi_app() -> _ConnectorASGIApp:
    """Build the combined ASGI app serving both connector services plus health checks."""
    async_app = AsyncConnectorServiceASGIApplication(ConnectAsyncConnectorService())
    metadata_app = ConnectorMetadataServiceASGIApplication(ConnectConnectorMetadataService())
    return _ConnectorASGIApp(
        applications=(
            (async_app.path, async_app),
            (metadata_app.path, metadata_app),
        )
    )
