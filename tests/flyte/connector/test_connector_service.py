import pathlib
import typing
from dataclasses import dataclass
from unittest.mock import MagicMock

import grpc
import pytest
from flyteidl.admin import agent_pb2
from flyteidl.admin.agent_pb2 import (
    CreateTaskRequest,
    DeleteTaskRequest,
    DeleteTaskResponse,
    GetTaskLogsRequest,
    GetTaskMetricsRequest,
    GetTaskRequest,
    Resource,
    TaskCategory,
)
from flyteidl.core import literals_pb2
from flyteidl.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl.core.identifier_pb2 import (
    Identifier,
    NodeExecutionIdentifier,
    ResourceType,
    TaskExecutionIdentifier,
    WorkflowExecutionIdentifier,
)
from flyteidl.core.literals_pb2 import LiteralMap
from flyteidl.core.security_pb2 import Identity
from flyteidl.core.tasks_pb2 import TaskTemplate

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.connectors._connector import AsyncConnector, ConnectorRegistry, FlyteConnectorNotFound, ResourceMeta
from flyte.connectors._server import AsyncConnectorService
from flyte.models import SerializationContext


@dataclass
class DummyMetadata(ResourceMeta):
    job_id: str
    output_path: typing.Optional[str] = None
    task_name: typing.Optional[str] = None


class DummyConnector(AsyncConnector):
    name = "Dummy Connector"

    def __init__(self):
        super().__init__(task_type_name="dummy", metadata_type=DummyMetadata)

    def create(self, task_template: TaskTemplate, inputs: typing.Optional[LiteralMap], **kwargs) -> DummyMetadata:
        return DummyMetadata(job_id="job_id")

    def get(self, resource_meta: DummyMetadata, **kwargs) -> Resource:
        return Resource(
            phase=TaskExecution.SUCCEEDED,
            log_links=[TaskLog(name="console", uri="localhost:3000")],
            custom_info={"custom": "info", "num": 1},
        )

    def delete(self, resource_meta: DummyMetadata, **kwargs): ...


def get_task_template() -> TaskTemplate:
    env = flyte.TaskEnvironment(
        name="test_env_cache",
        image="python:3.10",
        resources=flyte.Resources(cpu="1", memory="2Gi"),
        env_vars={"ENV1": "val1", "ENV2": "val2"},
    )

    # Create a task using the environment
    @env.task(
        short_name="real_test_task",
        cache=flyte.Cache(behavior="auto", ignored_inputs="my_ignored_input"),
        retries=3,
        timeout=60,
    )
    async def test_function(a: int, my_ignored_input: str) -> str:
        """Test function docstring"""
        return f"{a} {my_ignored_input}"

    # Get the task template from the decorated function
    task_template = test_function

    # Create serialization context
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    return get_proto_task(task_template, context)


def get_task_execution_metadata():
    return agent_pb2.TaskExecutionMetadata(
        task_execution_id=TaskExecutionIdentifier(
            task_id=Identifier(ResourceType.TASK, "project", "domain", "name", "version"),
            node_execution_id=NodeExecutionIdentifier(
                "node_id", WorkflowExecutionIdentifier("project", "domain", "name")
            ),
            retry_attempt=1,
        ),
        namespace="namespace",
        labels={"label_key": "label_val"},
        annotations={"annotation_key": "annotation_val"},
        k8s_service_account="k8s service account",
        environment_variables={"env_var_key": "env_var_val"},
        identity=Identity(execution_identity="task executor"),
    )


async def test_async_connector_service():
    connector = DummyConnector
    ConnectorRegistry.register(connector, override=True)
    service = AsyncConnectorService()
    ctx = MagicMock(spec=grpc.ServicerContext)

    inputs_proto = literals_pb2.LiteralMap(
        {
            "a": literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=1))),
        },
    )

    output_prefix = "/tmp"
    dummy_id = "dummy_id"
    metadata_bytes = (
        DummyMetadata(
            job_id=dummy_id,
            output_path=f"{output_prefix}/{dummy_id}",
            task_name=agent_pb2.TaskExecutionMetadata.task_execution_id.task_id.name,
        ).encode()
    )

    tmp = get_task_template()
    task_category = TaskCategory(name=connector.task_category.name, version=0)
    req = CreateTaskRequest(
        inputs=inputs_proto,
        template=tmp,
        output_prefix=output_prefix,
        task_execution_metadata=get_task_execution_metadata(),
    )

    res = await service.CreateTask(req, ctx)
    assert res.resource_meta == metadata_bytes
    res = await service.GetTask(GetTaskRequest(task_category=task_category, resource_meta=metadata_bytes), ctx)
    assert res.resource.phase == TaskExecution.SUCCEEDED
    res = await service.DeleteTask(
        DeleteTaskRequest(task_category=task_category, resource_meta=metadata_bytes),
        ctx,
    )
    assert res == DeleteTaskResponse()
    if connector.task_category.name == "async_dummy":
        res = await service.GetTaskMetrics(
            GetTaskMetricsRequest(task_category=task_category, resource_meta=metadata_bytes), ctx
        )
        assert res.results[0].metric == "EXECUTION_METRIC_LIMIT_MEMORY_BYTES"

        res = await service.GetTaskLogs(
            GetTaskLogsRequest(task_category=task_category, resource_meta=metadata_bytes), ctx
        )
        assert res.body.results == ["foo", "bar"]

    connector_metadata = ConnectorRegistry.get_connector_metadata(connector.name)
    assert connector_metadata.supported_task_types[0] == connector.task_category.name
    assert connector_metadata.supported_task_categories[0].name == connector.task_category.name

    with pytest.raises(FlyteConnectorNotFound):
        ConnectorRegistry.get_connector_metadata("non-exist-namr")
