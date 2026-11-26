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


class SilentAfterSentinelOnce:
    """A watch whose connection died after the sentinel: no error, no EOF, nothing
    more. The child finished meanwhile, so the next watch's snapshot has it."""

    def __init__(self, sentinel_delay: float = 0.0):
        self.watches = 0
        self.sentinel_delay = sentinel_delay
        self.first_sentinel_watch = None

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        if self.watches > 1:
            yield _succeeded("child")
        await asyncio.sleep(self.sentinel_delay)
        if self.first_sentinel_watch is None:
            self.first_sentinel_watch = self.watches
        yield _sentinel()
        await asyncio.Event().wait()


def _informer(service, **kwargs) -> Informer:
    return Informer(run_id=RUN, parent_action_name="a0", shared_queue=asyncio.Queue(), actions_client=service, **kwargs)


@pytest.mark.asyncio
async def test_a_silent_watch_reconnects_and_resyncs_the_children():
    """A watch silent past the idle timeout reconnects, and the new snapshot
    delivers the child that finished meanwhile."""
    service = SilentAfterSentinelOnce()
    # A long backoff shows the idle reconnect isn't counted as a failure.
    informer = _informer(service, watch_idle_timeout_sec=0.05, min_watch_backoff=30.0, max_watch_backoff=30.0)
    await informer.start(timeout=5)
    try:
        node = await asyncio.wait_for(informer._shared_queue.get(), timeout=5)
        assert node.name == "child"
        assert service.watches == 2
    finally:
        await informer.stop()


@pytest.mark.asyncio
async def test_the_idle_timeout_waits_for_the_sentinel():
    service = SilentAfterSentinelOnce(sentinel_delay=0.2)
    informer = _informer(service, watch_idle_timeout_sec=0.05)
    await informer.start(timeout=5)
    try:
        # A deadline before the sentinel would have cut the first watch short.
        assert service.first_sentinel_watch == 1
    finally:
        await informer.stop()


class EndedByTheServerOnce:
    """A watch the server ends cleanly after the sentinel; the next one has the finished child."""

    def __init__(self):
        self.watches = 0

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        if self.watches > 1:
            yield _succeeded("child")
        yield _sentinel()
        if self.watches > 1:
            await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_watch_the_server_ends_reconnects():
    service = EndedByTheServerOnce()
    informer = _informer(service, min_watch_backoff=0.01, max_watch_backoff=0.01)
    await informer.start(timeout=5)
    try:
        node = await asyncio.wait_for(informer._shared_queue.get(), timeout=5)
        assert node.name == "child"
        assert service.watches == 2
    finally:
        await informer.stop()


class Chatty:
    """A watch that keeps delivering updates, each inside the idle timeout."""

    def __init__(self, children: int):
        self.watches = 0
        self.children = children

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        yield _sentinel()
        for i in range(self.children):
            await asyncio.sleep(0.03)
            yield _succeeded(f"child{i}")
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_watch_that_keeps_delivering_is_not_reconnected():
    service = Chatty(children=5)
    informer = _informer(service, watch_idle_timeout_sec=0.1)
    await informer.start(timeout=5)
    try:
        for _ in range(5):
            await asyncio.wait_for(informer._shared_queue.get(), timeout=5)
        assert service.watches == 1
    finally:
        await informer.stop()


class DroppedAfterEverySnapshot:
    """Every watch delivers its snapshot and is then dropped, as when a proxy keeps
    resetting the connection under it. `up_for` is how long each watch stays open,
    and `live_update` sends one new update after the snapshot before the drop."""

    def __init__(self, up_for: float = 0.0, live_update: bool = False):
        self.watches = 0
        self.up_for = up_for
        self.live_update = live_update

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        yield _sentinel()
        if self.live_update:
            yield _succeeded(f"child{self.watches}")
        await asyncio.sleep(self.up_for)
        raise ConnectError(Code.CANCELED, "upstream reset")


@pytest.mark.asyncio
async def test_a_watch_dropped_after_every_snapshot_gives_up():
    """Before: each snapshot reset the failure count, so the watch reconnected
    every second or two forever and max_watch_retries never applied."""
    service = DroppedAfterEverySnapshot()
    informer = _informer(service, min_watch_backoff=0.001, max_watch_backoff=0.002, max_watch_retries=5)
    task = await informer.start(timeout=5)
    try:
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        assert isinstance(task.exception(), ConnectError)
        assert service.watches == 5
    finally:
        await informer.stop()


@pytest.mark.asyncio
async def test_a_watch_that_stays_up_between_drops_keeps_reconnecting():
    """A watch that stayed open past watch_healthy_after_sec was working, so its
    drop starts a new count instead of adding to the old one."""
    service = DroppedAfterEverySnapshot(up_for=0.03)
    informer = _informer(
        service,
        min_watch_backoff=0.001,
        max_watch_backoff=0.002,
        max_watch_retries=3,
        watch_healthy_after_sec=0.02,
    )
    task = await informer.start(timeout=5)
    try:
        while service.watches < 8:
            await asyncio.sleep(0.01)
            assert not task.done()
    finally:
        await informer.stop()


@pytest.mark.asyncio
async def test_a_watch_that_delivers_between_drops_keeps_reconnecting():
    """A watch dropped right after a new update was still working: each drop starts a new count."""
    service = DroppedAfterEverySnapshot(live_update=True)
    informer = _informer(service, min_watch_backoff=0.001, max_watch_backoff=0.002, max_watch_retries=3)
    task = await informer.start(timeout=5)
    try:
        while service.watches < 8:
            await asyncio.sleep(0.01)
            assert not task.done()
    finally:
        await informer.stop()


class HangsThenFails:
    """Every watch sends nothing for `hang` seconds and then fails."""

    def __init__(self, hang: float):
        self.watches = 0
        self.hang = hang

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        await asyncio.sleep(self.hang)
        raise ConnectError(Code.UNAVAILABLE, "connect failed")
        yield  # makes this an async generator


@pytest.mark.asyncio
async def test_a_watch_that_hangs_then_fails_gives_up():
    """Time spent before the first response isn't time the watch was up, so a
    connection that hangs past watch_healthy_after_sec and then fails still counts."""
    service = HangsThenFails(hang=0.03)
    informer = _informer(
        service,
        min_watch_backoff=0.001,
        max_watch_backoff=0.002,
        max_watch_retries=3,
        watch_healthy_after_sec=0.02,
    )
    task = await informer.start(timeout=0.01)
    try:
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        assert isinstance(task.exception(), ConnectError)
        assert service.watches == 3
    finally:
        await informer.stop()


class DroppedThenHealthy:
    """Odd watches are dropped right after their snapshot. Even watches stay up
    past watch_healthy_after_sec and then end: cleanly, or by going idle."""

    def __init__(self, up_for: float, ends: str):
        self.watches = 0
        self.up_for = up_for
        self.ends = ends

    async def watch_for_updates(self, req, **kwargs):
        self.watches += 1
        yield _sentinel()
        if self.watches % 2:
            raise ConnectError(Code.CANCELED, "upstream reset")
        if self.ends == "idle":
            await asyncio.Event().wait()
        await asyncio.sleep(self.up_for)


@pytest.mark.parametrize("ends", ["clean", "idle"])
@pytest.mark.asyncio
async def test_a_healthy_watch_resets_the_count_however_it_ends(ends):
    """Drops separated by a healthy watch are not back to back, so they never add
    up to max_watch_retries, whether the healthy watch ends cleanly or idles out."""
    service = DroppedThenHealthy(up_for=0.03, ends=ends)
    informer = _informer(
        service,
        min_watch_backoff=0.001,
        max_watch_backoff=0.002,
        max_watch_retries=2,
        watch_healthy_after_sec=0.02,
        watch_idle_timeout_sec=0.03,
    )
    task = await informer.start(timeout=5)
    try:
        while service.watches < 8:
            await asyncio.sleep(0.01)
            assert not task.done()
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
