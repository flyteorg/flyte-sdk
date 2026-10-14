"""listall's attr filters: what actually goes on the wire.

Filtering is server-side, so what matters is the Filter messages the request
carries -- a wrong field name or function silently returns the wrong set rather
than failing, which is why these assert the request rather than the results.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_service_pb2
from flyteidl2.common import list_pb2

from flyte.artifacts import KIND_KEY
from flyte.remote import Artifact
from flyte.remote._artifact import METADATA_FIELD_PREFIX


async def _captured_filters(**kwargs) -> list[list_pb2.Filter]:
    """Run listall against a stubbed client and return the filters it sent."""
    captured: list[list_pb2.Filter] = []

    async def fake_list(request):
        captured.extend(request.request.filters)
        return artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")

    client = MagicMock()
    client.artifact_service.list_artifacts = AsyncMock(side_effect=fake_list)

    with (
        patch("flyte.remote._artifact.ensure_client"),
        patch(
            "flyte.remote._artifact.get_init_config",
            return_value=MagicMock(org="test-org", project="proj", domain="dev"),
        ),
        patch("flyte.remote._artifact.get_client", return_value=client),
    ):
        async for _ in Artifact.listall.aio(**kwargs):
            pass
    return captured


def _by_field(filters, field):
    return next((f for f in filters if f.field == field), None)


@pytest.mark.asyncio
async def test_kind_becomes_a_reserved_key_filter():
    filters = await _captured_filters(kind="model")

    got = _by_field(filters, f"{METADATA_FIELD_PREFIX}{KIND_KEY}")
    assert got is not None, "kind= must filter on the reserved attr key"
    assert got.function == list_pb2.Filter.VALUE_IN
    assert list(got.values) == ["model"]


@pytest.mark.asyncio
async def test_attrs_single_value():
    filters = await _captured_filters(attrs={"framework": "torch"})

    got = _by_field(filters, f"{METADATA_FIELD_PREFIX}framework")
    assert got is not None
    assert list(got.values) == ["torch"]


@pytest.mark.asyncio
async def test_attrs_sequence_becomes_one_filter():
    """Values for one key are ORed by the server, so they ride in one filter."""
    filters = await _captured_filters(attrs={"framework": ["torch", "sklearn"]})

    got = _by_field(filters, f"{METADATA_FIELD_PREFIX}framework")
    assert list(got.values) == ["torch", "sklearn"]


@pytest.mark.asyncio
async def test_separate_keys_are_separate_filters():
    """Distinct keys must all match, which the server expresses as separate ANDed
    filters rather than one combined predicate."""
    filters = await _captured_filters(attrs={"framework": "torch", "team": "ml"})

    assert _by_field(filters, f"{METADATA_FIELD_PREFIX}framework") is not None
    assert _by_field(filters, f"{METADATA_FIELD_PREFIX}team") is not None


@pytest.mark.asyncio
async def test_kind_and_attrs_combine():
    """Both land in the same attr namespace; kind is not a separate mechanism."""
    filters = await _captured_filters(kind="model", attrs={"framework": "torch"})

    assert _by_field(filters, f"{METADATA_FIELD_PREFIX}{KIND_KEY}") is not None
    assert _by_field(filters, f"{METADATA_FIELD_PREFIX}framework") is not None


@pytest.mark.asyncio
async def test_empty_values_send_no_filter():
    """An empty sequence is not a predicate matching nothing; it is no predicate."""
    filters = await _captured_filters(attrs={"framework": []})

    assert _by_field(filters, f"{METADATA_FIELD_PREFIX}framework") is None


@pytest.mark.asyncio
async def test_no_filters_by_default():
    assert await _captured_filters() == []
