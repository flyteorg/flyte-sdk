"""Tests for Trigger.get_details, which fetches (and caches) a trigger's TriggerDetails."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.trigger import trigger_definition_pb2, trigger_service_pb2

from flyte.remote._trigger import Trigger, TriggerDetails


def _trigger() -> Trigger:
    pb2 = trigger_definition_pb2.Trigger(active=True)
    pb2.id.name.name = "t"
    pb2.id.name.task_name = "my_task"
    return Trigger(pb2=pb2)


def _details_response() -> trigger_service_pb2.GetTriggerDetailsResponse:
    return trigger_service_pb2.GetTriggerDetailsResponse(
        trigger=trigger_definition_pb2.TriggerDetails(
            id=identifier_pb2.TriggerIdentifier(
                name=identifier_pb2.TriggerName(name="t", task_name="my_task", org="o", project="p", domain="d")
            )
        )
    )


@contextmanager
def _mocked_client(client: MagicMock):
    cfg = MagicMock(org="o", project="p", domain="d")
    with (
        patch("flyte.remote._trigger.ensure_client"),
        patch("flyte.remote._trigger.get_init_config", return_value=cfg),
        patch("flyte.remote._trigger.get_client", return_value=client),
    ):
        yield


@pytest.mark.asyncio
async def test_get_details_requests_the_trigger_by_name_and_task_name():
    client = MagicMock()
    client.trigger_service.get_trigger_details = AsyncMock(return_value=_details_response())

    with _mocked_client(client):
        details = await _trigger().get_details()

    assert isinstance(details, TriggerDetails)
    assert details.name == "t"

    # Both halves of the trigger's identity have to be sent; the name alone does not identify it.
    req = client.trigger_service.get_trigger_details.await_args.kwargs["request"]
    assert req.name.name == "t"
    assert req.name.task_name == "my_task"
    assert (req.name.org, req.name.project, req.name.domain) == ("o", "p", "d")


@pytest.mark.asyncio
async def test_get_details_caches_the_fetched_details():
    client = MagicMock()
    client.trigger_service.get_trigger_details = AsyncMock(return_value=_details_response())

    trigger = _trigger()
    with _mocked_client(client):
        first = await trigger.get_details()
        second = await trigger.get_details()

    assert first is second is trigger.details
    client.trigger_service.get_trigger_details.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_details_returns_preloaded_details_without_a_request():
    client = MagicMock()
    client.trigger_service.get_trigger_details = AsyncMock(return_value=_details_response())

    preloaded = TriggerDetails(pb2=_details_response().trigger)
    trigger = _trigger()
    trigger.details = preloaded

    with _mocked_client(client):
        assert await trigger.get_details() is preloaded

    client.trigger_service.get_trigger_details.assert_not_awaited()
