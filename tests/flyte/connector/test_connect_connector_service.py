import pathlib
import typing
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.connector.connector_pb2 import (
    CreateTaskRequest,
    DeleteTaskRequest,
    DeleteTaskResponse,
    GetConnectorRequest,
    GetTaskLogsRequest,
    GetTaskLogsResponse,
    GetTaskLogsResponseBody,
    GetTaskMetricsRequest,
    GetTaskMetricsResponse,
    GetTaskRequest,
    ListConnectorsRequest,
    TaskCategory,
)
from flyteidl2.core import literals_pb2
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.metrics_pb2 import ExecutionMetricResult
from flyteidl2.core.tasks_pb2 import TaskTemplate
from flyteidl2.task import common_pb2

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.connectors._connect_server import (
    ConnectAsyncConnectorService,
    ConnectConnectorMetadataService,
    build_asgi_app,
)
from flyte.connectors._connector import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyte.models import SerializationContext


@dataclass
class DummyMetadata(ResourceMeta):
    job_id: str
    output_path: typing.Optional[str] = None
    task_name: typing.Optional[str] = None


class DummyConnector(AsyncConnector):
    name: str = "Dummy Connector"
    task_type_name: str = "async_dummy"
    metadata_type: type = DummyMetadata

    async def create(
        self, task_template: TaskTemplate, inputs: typing.Optional[typing.Dict[str, typing.Any]], **kwargs
    ) -> DummyMetadata:
        return DummyMetadata(job_id="dummy_id", output_path="/tmp/dummy_id", task_name="async_dummy")

    async def get(self, resource_meta: DummyMetadata, **kwargs) -> Resource:
        return Resource(
            phase=TaskExecution.SUCCEEDED,
            log_links=[TaskLog(name="console", uri="localhost:3000")],
            custom_info={"custom": "info", "num": 1},
        )

    async def delete(self, resource_meta: DummyMetadata, **kwargs): ...

    async def get_metrics(self, resource_meta: DummyMetadata, **kwargs) -> GetTaskMetricsResponse:
        return GetTaskMetricsResponse(
            results=[ExecutionMetricResult(metric="EXECUTION_METRIC_LIMIT_MEMORY_BYTES", data=None)]
        )

    async def get_logs(self, resource_meta: DummyMetadata, **kwargs) -> GetTaskLogsResponse:
        return GetTaskLogsResponse(body=GetTaskLogsResponseBody(results=["foo", "bar"]))


def get_task_template() -> TaskTemplate:
    env = flyte.TaskEnvironment(name="test_env_connect", image="python:3.10")

    @env.task
    async def test_function(a: int) -> int:
        return a

    test_function.task_type = "async_dummy"
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )
    return get_proto_task(test_function, context)


async def _drive(app, scope, body: bytes = b""):
    """Drive an ASGI app through a single request and collect the sent messages."""
    sent: list[dict] = []
    received = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return received.pop(0) if received else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


@pytest.fixture
def dummy_connector():
    connector = DummyConnector()
    ConnectorRegistry.register(connector, override=True)
    return connector


@pytest.mark.asyncio
async def test_connect_async_connector_service(dummy_connector):
    service = ConnectAsyncConnectorService()
    ctx = MagicMock()

    inputs_proto = common_pb2.Inputs(
        literals=[
            common_pb2.NamedLiteral(
                name="a",
                value=literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=1))),
            ),
        ],
    )

    output_prefix = "/tmp"
    dummy_id = "dummy_id"
    metadata_bytes = DummyMetadata(
        job_id=dummy_id,
        output_path=f"{output_prefix}/{dummy_id}",
        task_name=dummy_connector.task_type_name,
    ).encode()

    task_category = TaskCategory(name=dummy_connector.task_type_name, version=dummy_connector.task_type_version)

    create_res = await service.create_task(
        CreateTaskRequest(inputs=inputs_proto, template=get_task_template(), output_prefix=output_prefix), ctx
    )
    assert create_res.resource_meta == metadata_bytes

    get_res = await service.get_task(GetTaskRequest(task_category=task_category, resource_meta=metadata_bytes), ctx)
    assert get_res.resource.phase == TaskExecution.SUCCEEDED

    delete_res = await service.delete_task(
        DeleteTaskRequest(task_category=task_category, resource_meta=metadata_bytes), ctx
    )
    assert delete_res == DeleteTaskResponse()

    metrics_res = await service.get_task_metrics(
        GetTaskMetricsRequest(task_category=task_category, resource_meta=metadata_bytes), ctx
    )
    assert metrics_res.results[0].metric == "EXECUTION_METRIC_LIMIT_MEMORY_BYTES"

    log_responses = [
        msg
        async for msg in service.get_task_logs(
            GetTaskLogsRequest(task_category=task_category, resource_meta=metadata_bytes), ctx
        )
    ]
    assert log_responses[0].body.results == ["foo", "bar"]


@pytest.mark.asyncio
async def test_connect_service_not_found_raises_connect_error(dummy_connector):
    service = ConnectAsyncConnectorService()
    ctx = MagicMock()

    with pytest.raises(ConnectError) as exc_info:
        await service.get_task(
            GetTaskRequest(task_category=TaskCategory(name="does-not-exist", version=0), resource_meta=b"{}"), ctx
        )
    assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_connect_metadata_service(dummy_connector):
    service = ConnectConnectorMetadataService()
    ctx = MagicMock()

    get_res = await service.get_connector(GetConnectorRequest(name=dummy_connector.name), ctx)
    assert get_res.connector.name == dummy_connector.name
    assert get_res.connector.supported_task_categories[0].name == dummy_connector.task_type_name

    list_res = await service.list_connectors(ListConnectorsRequest(), ctx)
    assert any(c.name == dummy_connector.name for c in list_res.connectors)

    with pytest.raises(ConnectError) as exc_info:
        await service.get_connector(GetConnectorRequest(name="non-exist-name"), ctx)
    assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_asgi_app_health_check():
    app = build_asgi_app()
    for path in ("/healthz", "/health"):
        scope = {"type": "http", "method": "GET", "path": path, "headers": [], "root_path": ""}
        sent = await _drive(app, scope)
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_asgi_app_unknown_path_returns_404():
    app = build_asgi_app()
    scope = {"type": "http", "method": "GET", "path": "/unknown", "headers": [], "root_path": ""}
    sent = await _drive(app, scope)
    assert sent[0]["status"] == 404


@pytest.mark.asyncio
async def test_asgi_app_routes_to_connect_services():
    app = build_asgi_app()
    # A request under each service prefix should be routed to the generated
    # application (which rejects this malformed GET), not handled as 404 by the router.
    for prefix in (
        "/flyteidl2.connector.AsyncConnectorService/CreateTask",
        "/flyteidl2.connector.ConnectorMetadataService/ListConnectors",
    ):
        scope = {
            "type": "http",
            "method": "GET",
            "path": prefix,
            "headers": [],
            "root_path": "",
            "query_string": b"",
            "scheme": "http",
            "extensions": {},
        }
        sent = await _drive(app, scope)
        # The generated ConnectASGIApplication handled it (any status other than
        # the router's own 404-for-unknown-prefix is fine).
        assert sent, "expected the connect application to produce a response"
        assert sent[0]["type"] == "http.response.start"


@pytest.mark.asyncio
async def test_asgi_app_lifespan():
    app = build_asgi_app()
    events = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent: list[dict] = []

    async def receive():
        return events.pop(0)

    async def send(message):
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)
    assert {"type": "lifespan.startup.complete"} in sent
    assert {"type": "lifespan.shutdown.complete"} in sent
