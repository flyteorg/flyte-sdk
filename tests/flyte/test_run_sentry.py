"""The SDK counts create_run so it can be compared against the same operation
counted server-side by the Flyte backend."""

import mock
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
import flyte.errors
from flyte._initialize import _init_for_testing
from flyte.models import CodeBundle

env = flyte.TaskEnvironment(name="test")


@env.task
async def task1(v: str) -> str:
    return f"Hello, world {v}!"


def _make_mock_client():
    mock_client = MagicMock()
    mock_client.run_service = AsyncMock()

    mock_dataproxy_service = AsyncMock()
    mock_dataproxy_service.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=common_run_pb2.OffloadedInputData(uri="s3://bucket/inputs", inputs_hash="abc123"),
    )
    mock_client.dataproxy_service = mock_dataproxy_service
    return mock_client


def _patch_build(fn):
    fn = mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)(fn)
    fn = mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)(fn)
    return fn


async def _run_remote(mock_build_image_bg, mock_code_bundler, mock_client):
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)
    await _init_for_testing(client=mock_client, project="test", domain="test")
    return await flyte.with_runcontext(mode="remote").run.aio(task1, "hello")


def _operation_calls(count_mock):
    """The (key, tags) of every flyte.operation counter emitted."""
    return [
        (call.args[0], call.kwargs.get("tags"))
        for call in count_mock.call_args_list
        if call.args and call.args[0] == "flyte.operation"
    ]


@pytest.mark.asyncio
@_patch_build
async def test_create_run_counts_success(mock_code_bundler, mock_build_image_bg):
    with mock.patch("flyte._sentry.count") as count_mock:
        await _run_remote(mock_build_image_bg, mock_code_bundler, _make_mock_client())

    assert _operation_calls(count_mock) == [("flyte.operation", {"operation": "create_run", "status": "success"})]


@pytest.mark.asyncio
@_patch_build
async def test_create_run_counts_failure(mock_code_bundler, mock_build_image_bg):
    mock_client = _make_mock_client()
    mock_client.run_service.create_run.side_effect = ConnectError(Code.INVALID_ARGUMENT, "bad inputs")

    with mock.patch("flyte._sentry.count") as count_mock:
        with pytest.raises(flyte.errors.RuntimeUserError):
            await _run_remote(mock_build_image_bg, mock_code_bundler, mock_client)

    calls = _operation_calls(count_mock)
    assert len(calls) == 1
    tags = calls[0][1]
    assert tags["operation"] == "create_run"
    assert tags["status"] == "error"
    # An invalid argument is the caller's fault, not a backend bug.
    assert tags["error_kind"] == "user"
