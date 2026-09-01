"""The root action's cache key must honor `Literal.hash`, like a sub-action's does.

`_submit_remote` takes `OffloadedInputData.inputs_hash` from the server, which derives it from
the marshaled inputs — folding in the offloaded blob URI and ignoring `Literal.hash`. That made
content-based caching degrade to URI-based caching at the run entrypoint: re-running with
byte-identical content uploaded to a fresh URI missed, while the same task invoked as a
sub-action hit. These tests pin the client-side override that closes the gap.
"""

from types import SimpleNamespace

import mock
import pytest
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.core import identifier_pb2, interface_pb2, literals_pb2, tasks_pb2, types_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2 as task_common_pb2
from flyteidl2.task import run_pb2, task_definition_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
from flyte._initialize import _init_for_testing

SERVER_HASH = "server-derived-hash"
CONTENT_HASH = "sha256-of-geoparquet-bytes"


def _sd_inputs(uri: str, hash_val: str | None = None) -> task_common_pb2.Inputs:
    """One StructuredDataset input, optionally carrying a user-supplied content hash."""
    return task_common_pb2.Inputs(
        literals=[
            task_common_pb2.NamedLiteral(
                name="aoi",
                value=literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(
                        structured_dataset=literals_pb2.StructuredDataset(
                            uri=uri,
                            metadata=literals_pb2.StructuredDatasetMetadata(
                                structured_dataset_type=types_pb2.StructuredDatasetType(format="parquet")
                            ),
                        )
                    ),
                    hash=hash_val,
                ),
            )
        ]
    )


def _mock_client(prior_inputs: task_common_pb2.Inputs):
    client = MagicMock()
    client.run_service = AsyncMock()

    dataproxy = AsyncMock()
    dataproxy.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=common_run_pb2.OffloadedInputData(uri="s3://b/inputs", inputs_hash=SERVER_HASH),
    )
    dataproxy.get_action_data.return_value = dataproxy_service_pb2.GetActionDataResponse(inputs=prior_inputs)
    client.dataproxy_service = dataproxy
    return client


def _fake_prior_run(ignored_input_vars=()):
    iface = interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(
                    key="aoi",
                    value=interface_pb2.Variable(
                        type=types_pb2.LiteralType(structured_dataset_type=types_pb2.StructuredDatasetType())
                    ),
                )
            ]
        )
    )
    task_spec = task_definition_pb2.TaskSpec(
        task_template=tasks_pb2.TaskTemplate(
            id=identifier_pb2.Identifier(name="test.validate_aoi", version="v1"),
            interface=iface,
            metadata=tasks_pb2.TaskMetadata(cache_ignore_input_vars=list(ignored_input_vars)),
        )
    )
    return SimpleNamespace(
        pb2=SimpleNamespace(run_spec=run_pb2.RunSpec()),
        action_details=SimpleNamespace(pb2=run_definition_pb2.ActionDetails(task=task_spec)),
    )


async def _submit(prior_inputs: task_common_pb2.Inputs, ignored_input_vars=()) -> run_service_pb2.CreateRunRequest:
    """Drive a submission with `prior_inputs` and hand back the CreateRunRequest sent."""
    client = _mock_client(prior_inputs)
    await _init_for_testing(client=client, project="test", domain="test")
    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run(ignored_input_vars))
        await flyte.with_runcontext(mode="remote").rerun.aio("r1")
    return client.run_service.create_run.call_args[0][0]


@pytest.mark.asyncio
async def test_hashed_inputs_override_the_server_inputs_hash():
    req = await _submit(_sd_inputs("s3://bkt/run-1/abc/0", CONTENT_HASH))
    assert req.offloaded_input_data.inputs_hash != SERVER_HASH
    # The blob reference itself is untouched — only the cache-key input changes.
    assert req.offloaded_input_data.uri == "s3://b/inputs"


@pytest.mark.asyncio
async def test_same_content_at_a_new_uri_yields_the_same_key():
    """The customer-reported miss: identical content, fresh upload URI."""
    first = await _submit(_sd_inputs("s3://bkt/run-1/abc/0", CONTENT_HASH))
    second = await _submit(_sd_inputs("s3://bkt/run-2/xyz/0", CONTENT_HASH))

    assert first.offloaded_input_data.inputs_hash == second.offloaded_input_data.inputs_hash


@pytest.mark.asyncio
async def test_changed_content_yields_a_different_key():
    first = await _submit(_sd_inputs("s3://bkt/run-1/abc/0", CONTENT_HASH))
    second = await _submit(_sd_inputs("s3://bkt/run-1/abc/0", "a-different-digest"))

    assert first.offloaded_input_data.inputs_hash != second.offloaded_input_data.inputs_hash


@pytest.mark.asyncio
async def test_unhashed_inputs_keep_the_server_value():
    """No content hash anywhere: defer to the backend, so existing cache entries stay reachable."""
    req = await _submit(_sd_inputs("s3://bkt/run-1/abc/0"))
    assert req.offloaded_input_data.inputs_hash == SERVER_HASH


def _with_seed(inputs: task_common_pb2.Inputs, seed: int) -> task_common_pb2.Inputs:
    """Append a plain int input, standing in for a cache-ignored one (e.g. a run counter)."""
    out = task_common_pb2.Inputs()
    out.CopyFrom(inputs)
    out.literals.append(
        task_common_pb2.NamedLiteral(
            name="seed",
            value=literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=seed))),
        )
    )
    return out


@pytest.mark.asyncio
async def test_cache_ignored_inputs_are_excluded_from_the_override():
    """The ignore list must come off the task spec, matching `filterInputsForHash` server-side.

    Without it, a task combining `Cache(ignored_inputs=...)` with a content-hashed input would
    key on the very inputs the user asked to exclude.
    """
    base = _sd_inputs("s3://bkt/run-1/abc/0", CONTENT_HASH)
    first = await _submit(_with_seed(base, 1), ignored_input_vars=["seed"])
    second = await _submit(_with_seed(base, 2), ignored_input_vars=["seed"])

    assert first.offloaded_input_data.inputs_hash == second.offloaded_input_data.inputs_hash


@pytest.mark.asyncio
async def test_without_the_ignore_list_the_extra_input_moves_the_key():
    """Guards the test above against passing for the wrong reason."""
    base = _sd_inputs("s3://bkt/run-1/abc/0", CONTENT_HASH)
    first = await _submit(_with_seed(base, 1))
    second = await _submit(_with_seed(base, 2))

    assert first.offloaded_input_data.inputs_hash != second.offloaded_input_data.inputs_hash
