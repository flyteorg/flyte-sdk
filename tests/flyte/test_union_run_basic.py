import mock
import pytest
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
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
    assert result.outputs()[0] == "Hello, world test!"


@pytest.mark.asyncio
async def test_task1_local_union_async():
    await flyte.init.aio()
    result = await flyte.run.aio(task1, "test")
    assert result.outputs()[0] == "Hello, world test!"


@pytest.mark.asyncio
@mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)
@mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)
@mock.patch("flyte.remote._client.controlplane.ClientSet")  # Patch the Client class
async def test_task1_remote_union_sync(
    mock_client_class: MagicMock, mock_code_bundler: AsyncMock, mock_build_image_bg: AsyncMock
):
    mock_client = mock_client_class.return_value  # Mocked client instance
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service  # Set the mocked run_service

    mock_dataproxy_service = AsyncMock()
    mock_offloaded = common_run_pb2.OffloadedInputData(uri="s3://bucket/inputs", inputs_hash="abc123")
    mock_dataproxy_service.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=mock_offloaded,
    )
    mock_client.dataproxy_service = mock_dataproxy_service

    inputs = "say test"

    mock_code_bundler.return_value = CodeBundle(
        computed_version="v1",
        tgz="test.tgz",
    )
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(
        client=mock_client,
        project="test",
        domain="test",
    )
    run = await flyte.with_runcontext(mode="remote").run.aio(task1, inputs)

    # Ensure the run is not None
    assert run
    # Ensure upload_inputs was called with the correct inputs
    mock_dataproxy_service.upload_inputs.assert_called_once()
    upload_req: dataproxy_service_pb2.UploadInputsRequest = mock_dataproxy_service.upload_inputs.call_args[0][0]
    assert upload_req.inputs == common_pb2.Inputs(
        literals=[
            common_pb2.NamedLiteral(
                name="v",
                value=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value="say test"))
                ),
            ),
        ]
    )
    assert upload_req.WhichOneof("id") == "project_id"
    assert upload_req.project_id.name == "test"
    assert upload_req.project_id.domain == "test"
    assert upload_req.WhichOneof("task") == "task_spec"
    assert upload_req.task_spec.task_template.id.name == "test.task1"

    # Ensure create_run uses offloaded_input_data instead of inline inputs
    mock_build_image_bg.assert_called_once()
    mock_run_service.create_run.assert_called_once()
    captured_input = mock_run_service.create_run.call_args[0]
    req: run_service_pb2.CreateRunRequest = captured_input[0]
    assert req.offloaded_input_data == mock_offloaded
    assert not req.HasField("inputs")
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
        "--run-start-time",
        "{{.runStartTime}}",
        "--image-cache",
        req.task_spec.task_template.container.args[20],  # Image cache is dynamic
        "--tgz",
        "test.tgz",
        "--dest",
        ".",
        "--resolver",
        "flyte._internal.resolvers.default.DefaultTaskResolver",
        "mod",
        req.task_spec.task_template.container.args[28],  # changes based on where you run this test from
        "instance",
        "task1",
    ]
    # The default image no longer defaults its *push* registry to ghcr.io/flyteorg (end users
    # cannot push there — it 403s), so a dev-build default image has no registry and its uri is
    # just <name>:<tag>. Assert the default image's identity instead of a registry it must not
    # carry. The released image still *pulls* from ghcr.io/flyteorg via its base_image.
    assert Image.from_debian_base().name == "flyte"


@pytest.mark.asyncio
@mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)
@mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)
@mock.patch("flyte.remote._client.controlplane.ClientSet")
async def test_upload_inputs_with_run_id(
    mock_client_class: MagicMock, mock_code_bundler: AsyncMock, mock_build_image_bg: AsyncMock
):
    """When a run name is provided, upload_inputs should use run_id."""
    mock_client = mock_client_class.return_value
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service

    mock_dataproxy_service = AsyncMock()
    mock_offloaded = common_run_pb2.OffloadedInputData(uri="s3://bucket/inputs", inputs_hash="key456")
    mock_dataproxy_service.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=mock_offloaded,
    )
    mock_client.dataproxy_service = mock_dataproxy_service

    mock_code_bundler.return_value = CodeBundle(
        computed_version="v1",
        tgz="test.tgz",
    )
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(
        client=mock_client,
        project="testproject",
        domain="development",
    )
    run = await flyte.with_runcontext(mode="remote", name="my-run").run.aio(task1, "hello")

    assert run

    # upload_inputs should use run_id when name is provided
    upload_req: dataproxy_service_pb2.UploadInputsRequest = mock_dataproxy_service.upload_inputs.call_args[0][0]
    assert upload_req.WhichOneof("id") == "run_id"
    assert upload_req.run_id.name == "my-run"
    assert upload_req.run_id.project == "testproject"
    assert upload_req.run_id.domain == "development"
    assert upload_req.WhichOneof("task") == "task_spec"
    assert upload_req.task_spec.task_template.id.name == "test.task1"

    # create_run should use offloaded_input_data
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.offloaded_input_data == mock_offloaded
    assert not req.HasField("inputs")


def _make_mock_client():
    """Build a mocked ClientSet with run + dataproxy services wired for create_run tests."""
    mock_client = MagicMock()
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service

    mock_dataproxy_service = AsyncMock()
    mock_offloaded = common_run_pb2.OffloadedInputData(uri="s3://bucket/inputs", inputs_hash="abc123")
    mock_dataproxy_service.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=mock_offloaded,
    )
    mock_client.dataproxy_service = mock_dataproxy_service
    return mock_client, mock_run_service


@pytest.mark.asyncio
@mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)
@mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)
async def test_run_spec_max_action_concurrency(mock_code_bundler: AsyncMock, mock_build_image_bg: AsyncMock):
    """max_action_concurrency from with_runcontext should land on RunSpec."""
    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(client=mock_client, project="test", domain="test")
    run = await flyte.with_runcontext(mode="remote", max_action_concurrency=5).run.aio(task1, "hello")

    assert run
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.max_action_concurrency == 5


@pytest.mark.asyncio
@mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)
@mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)
async def test_run_spec_max_action_concurrency_default_unset(
    mock_code_bundler: AsyncMock, mock_build_image_bg: AsyncMock
):
    """When max_action_concurrency is not provided, RunSpec carries the proto default (0 = unset)."""
    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(client=mock_client, project="test", domain="test")
    run = await flyte.with_runcontext(mode="remote").run.aio(task1, "hello")

    assert run
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    assert req.run_spec.max_action_concurrency == 0


def test_with_runcontext_rejects_negative_max_action_concurrency():
    with pytest.raises(ValueError, match="max_action_concurrency"):
        flyte.with_runcontext(max_action_concurrency=-1)


def test_with_runcontext_rejects_max_action_concurrency_of_one():
    """A value of 1 would deadlock: the parent action holds the only concurrency slot."""
    with pytest.raises(ValueError, match="deadlock"):
        flyte.with_runcontext(max_action_concurrency=1)


def test_with_runcontext_allows_zero_max_action_concurrency():
    flyte.with_runcontext(max_action_concurrency=0)
    flyte.with_runcontext(max_action_concurrency=2)
