"""Tests for Logs.tail asking the log plane for a connector's logs (e.g. a GPU task on a shared queue)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.core import execution_pb2
from flyteidl2.workflow import run_service_pb2

from flyte.remote._logs import Logs


def _action_id() -> identifier_pb2.ActionIdentifier:
    return identifier_pb2.ActionIdentifier(run=identifier_pb2.RunIdentifier(name="run-1"), name="a0")


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _client(log_context=None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    client.dataproxy_service.tail_logs = MagicMock(return_value=_EmptyStream())
    if error is not None:
        client.run_service.get_action_log_context = AsyncMock(side_effect=error)
    else:
        client.run_service.get_action_log_context = AsyncMock(
            return_value=run_service_pb2.GetActionLogContextResponse(log_context=log_context)
        )
    return client


async def _tail(client: MagicMock) -> str:
    with patch("flyte.remote._logs.ensure_client"), patch("flyte.remote._logs.get_client", return_value=client):
        _ = [line async for line in Logs.tail.aio(action_id=_action_id())]
    return client.dataproxy_service.tail_logs.call_args.args[0].connector_endpoint


@pytest.mark.asyncio
async def test_connector_action_tails_its_connector():
    log_context = execution_pb2.LogContext(connector=execution_pb2.ConnectorLogContext(endpoint="lw.ns.svc:8001"))
    assert await _tail(_client(log_context)) == "lw.ns.svc:8001"


@pytest.mark.asyncio
async def test_pod_action_tails_pod_logs():
    log_context = execution_pb2.LogContext(primary_pod_name="pod-a0")
    assert await _tail(_client(log_context)) == ""


@pytest.mark.asyncio
async def test_no_log_context_falls_back_to_pod_logs():
    assert await _tail(_client(error=RuntimeError("unavailable"))) == ""
