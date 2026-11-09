"""Failure handling in the remote controller: the informer's watch, reading a
child's outputs, and retry backoff."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.actions import actions_service_pb2
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.task import common_pb2
from flyteidl2.workflow import state_service_pb2

import flyte.errors
from flyte._internal.controllers.remote import _controller
from flyte._internal.controllers.remote._informer import Informer

RUN = identifier_pb2.RunIdentifier(org="org", project="proj", domain="dev", name="run1")


def _sentinel() -> actions_service_pb2.WatchForUpdatesResponse:
    return actions_service_pb2.WatchForUpdatesResponse(
        control_message=state_service_pb2.ControlMessage(sentinel=True),
    )


def _succeeded(name: str) -> actions_service_pb2.WatchForUpdatesResponse:
    return actions_service_pb2.WatchForUpdatesResponse(
        action_update=state_service_pb2.ActionUpdate(
            action_id=identifier_pb2.ActionIdentifier(name=name, run=RUN),
            phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
            output_uri="s3://bucket/run1/child",
        )
    )


class CancelledByTheServerOnce:
    """A watch the server (or a proxy in front of it) ends with CANCELED, as it
    does when a connection is dropped under it. The next watch delivers the
    child's update."""

    def __init__(self):
        self.watches = 0

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        yield _sentinel()
        if self.watches == 1:
            raise ConnectError(Code.CANCELED, "upstream reset")
        yield _succeeded("child")
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_watch_cancelled_by_the_server_reconnects():
    """Before: the informer took any CANCELED for its own stop and returned, so
    the watch ended silently and every waiter on a child hung."""
    service = CancelledByTheServerOnce()
    queue: asyncio.Queue = asyncio.Queue()
    informer = Informer(
        run_id=RUN,
        parent_action_name="a0",
        shared_queue=queue,
        actions_client=service,
        min_watch_backoff=0.01,
        max_watch_backoff=0.02,
    )
    await informer.start(timeout=5)
    try:
        node = await asyncio.wait_for(queue.get(), timeout=5)
        assert node.name == "child"
        assert service.watches == 2
        assert not informer.watch_task.done()
    finally:
        await informer.stop()


@pytest.mark.asyncio
async def test_stopping_the_informer_still_ends_its_watch():
    service = CancelledByTheServerOnce()
    informer = Informer(
        run_id=RUN,
        parent_action_name="a0",
        shared_queue=asyncio.Queue(),
        actions_client=service,
        min_watch_backoff=0.01,
        max_watch_backoff=0.02,
    )
    task = await informer.start(timeout=5)
    await informer.stop()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
    assert task.done()
    assert not informer.is_running()


def test_jittered_waits_between_half_and_all_of_the_delay():
    from flyte._utils.helpers import jittered

    waits = [jittered(8.0) for _ in range(1000)]
    assert all(4.0 <= w <= 8.0 for w in waits)
    # Spread out, so that clients that failed together do not retry together.
    assert max(waits) - min(waits) > 2.0


OUTPUTS = common_pb2.Outputs()


@pytest.mark.asyncio
async def test_a_transient_failure_reading_outputs_is_retried(monkeypatch):
    monkeypatch.setattr(_controller, "OUTPUT_READ_BACKOFF_SEC", 0.0, raising=False)
    with (
        patch(
            "flyte._internal.runtime.io.load_outputs",
            new_callable=AsyncMock,
            side_effect=[OSError("connection reset"), OSError("connection reset"), OUTPUTS],
        ) as load,
        patch("flyte._internal.runtime.convert.convert_outputs_to_native", new_callable=AsyncMock) as convert,
    ):
        convert.return_value = 3
        value = await _controller.load_and_convert_outputs(None, "s3://bucket/run1/child", max_bytes=100)
    assert value == 3
    assert load.await_count == 3


@pytest.mark.asyncio
async def test_outputs_that_cannot_be_read_fail_with_the_original_error(monkeypatch):
    """After a couple of tries, the failure stays what it was: a user error."""
    monkeypatch.setattr(_controller, "OUTPUT_READ_BACKOFF_SEC", 0.0, raising=False)
    with patch(
        "flyte._internal.runtime.io.load_outputs",
        new_callable=AsyncMock,
        side_effect=FileNotFoundError("no outputs.pb"),
    ) as load:
        with pytest.raises(FileNotFoundError):
            await _controller.load_and_convert_outputs(None, "s3://bucket/run1/child", max_bytes=100)
    assert load.await_count == getattr(_controller, "OUTPUT_READ_ATTEMPTS", 3)


@pytest.mark.asyncio
async def test_outputs_over_the_size_limit_are_not_retried(monkeypatch):
    monkeypatch.setattr(_controller, "OUTPUT_READ_BACKOFF_SEC", 0.0, raising=False)
    with patch(
        "flyte._internal.runtime.io.load_outputs",
        new_callable=AsyncMock,
        side_effect=flyte.errors.InlineIOMaxBytesBreached("too big"),
    ) as load:
        with pytest.raises(flyte.errors.InlineIOMaxBytesBreached):
            await _controller.load_and_convert_outputs(None, "s3://bucket/run1/child", max_bytes=100)
    assert load.await_count == 1
