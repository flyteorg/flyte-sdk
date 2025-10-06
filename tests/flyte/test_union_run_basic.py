import mock
import pytest
from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2
from flyteidl2.workflow import run_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
from flyte._image import Image
from flyte._initialize import _init_for_testing
from flyte.models import CodeBundle

env = flyte.TaskEnvironment(
    name="test",
)


@env.task
async def task1(v: str) -> str:
    return f"Hello, world {v}!"


@pytest.mark.asyncio
async def test_task1_local_direct():
    result = await task1("test")
    assert result == "Hello, world test!"


def test_task1_local_union_sync():
    flyte.init()
    result = flyte.run(task1, "test")
    assert result.outputs() == "Hello, world test!"


@pytest.mark.asyncio
async def test_task1_local_union_async():
    await flyte.init.aio()
    result = await flyte.run.aio(task1, "test")
    assert result.outputs() == "Hello, world test!"


@pytest.mark.asyncio
@mock.patch("flyte._deploy._build_image_bg")
@mock.patch("flyte._code_bundle.build_code_bundle")
@mock.patch("flyte.remote._client.controlplane.ClientSet")  # Patch the Client class
async def test_task1_remote_union_sync(
    mock_client_class: MagicMock, mock_code_bundler: AsyncMock, mock_build_image_bg: AsyncMock
):
    mock_client = mock_client_class.return_value  # Mocked client instance
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service  # Set the mocked run_service

    inputs = "say test"

    async def get_code_bundle():
        return CodeBundle(
            computed_version="v1",
            tgz="test.tgz",
        )

    mock_code_bundler.return_value = get_code_bundle()
    mock_build_image_bg.return_value = (env.name, "image_name")

    await _init_for_testing(
        client=mock_client,
        project="test",
        domain="test",
    )
    run = await flyte.with_runcontext(mode="remote").run.aio(task1, inputs)

    # Ensure the run is not None
    assert run
    # Ensure the mocked run_service.CreateRun is called
    mock_build_image_bg.assert_called_once()
    mock_run_service.CreateRun.assert_called_once()
    captured_input = mock_run_service.CreateRun.call_args[0]
    req: run_service_pb2.CreateRunRequest = captured_input[0]
    assert req.inputs == common_pb2.Inputs(
        literals=[
            common_pb2.NamedLiteral(
                name="v",
                value=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value="say test"))
                ),
            ),
        ]
    )
    assert req.project_id.name == "test"
    assert req.project_id.domain == "test"
    assert req.task_spec is not None
    assert req.task_spec.task_template.id.name == "test.task1"

    assert req.task_spec.task_template.container
    assert req.task_spec.task_template.container.args == [
        "a0",
        "--inputs",
        "{{.input}}",
        "--outputs-path",
        "{{.outputPrefix}}",
        "--version",
        "v1",
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
        "--image-cache",
        req.task_spec.task_template.container.args[18],  # Image cache is dynamic
        "--tgz",
        "test.tgz",
        "--dest",
        ".",
        "--resolver",
        "flyte._internal.resolvers.default.DefaultTaskResolver",
        "mod",
        req.task_spec.task_template.container.args[26],  # changes based on where you run this test from
        "instance",
        "task1",
    ]
    assert "ghcr.io/flyteorg/flyte" in Image.from_debian_base().uri
