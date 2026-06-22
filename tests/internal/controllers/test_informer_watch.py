"""
Regression tests for Informer.watch() stream recovery.

Incident: a flyte2 server restart left the parent action's watch stream wedged
forever. The runtime reconnected onto a pooled keep-alive connection that was
half-open (peer gone, no RST), wrote the WatchForUpdates request, and then
blocked indefinitely reading the chunked response because the transport's
read_timeout is unset and the intended `watch_conn_timeout_sec` guard was never
wired in. The parent never re-synced its children's terminal states and stayed
"Running" for hours.
"""

import asyncio

import pytest
from flyteidl2.actions import actions_service_pb2
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.workflow import state_service_pb2

from flyte._internal.controllers.remote._informer import Informer


def _update(child_id, phase, output_uri="s3://out"):
    return actions_service_pb2.WatchForUpdatesResponse(
        action_update=state_service_pb2.ActionUpdate(
            action_id=child_id, phase=phase, output_uri=output_uri
        )
    )


def _sentinel():
    return actions_service_pb2.WatchForUpdatesResponse(
        control_message=state_service_pb2.ControlMessage(sentinel=True)
    )


class _FlakyActions:
    """watch_for_updates that establishes once, then hangs on reconnect, then heals."""

    def __init__(self, child_id):
        self._child_id = child_id
        self.calls = 0

    async def watch_for_updates(self, req, **kwargs):
        self.calls += 1
        n = self.calls
        if n == 1:
            # Initial establishment succeeds (snapshot + sentinel), then the
            # stream dies -> informer is now "ready" and observed the child RUNNING.
            yield _update(self._child_id, phase_pb2.ACTION_PHASE_RUNNING)
            yield _sentinel()
            return
        if n == 2:
            # Reconnect onto a half-open pooled connection: the request is sent
            # but the snapshot read never returns. Without an establishment
            # deadline this wedges the watch (and the parent) forever.
            await asyncio.sleep(3600)
            yield  # pragma: no cover
            return
        # Healthy reconnect: server replays the snapshot, child now terminal.
        yield _update(self._child_id, phase_pb2.ACTION_PHASE_SUCCEEDED)
        yield _sentinel()
        await asyncio.sleep(3600)  # keep the stream open, idle


@pytest.mark.asyncio
async def test_informer_recovers_from_hung_reconnect():
    run_id = identifier_pb2.RunIdentifier(name="r1", project="p", domain="d")
    child_id = identifier_pb2.ActionIdentifier(name="a1", run=run_id)
    flaky = _FlakyActions(child_id)

    informer = Informer(
        run_id=run_id,
        parent_action_name="a0",
        shared_queue=asyncio.Queue(),
        actions_client=flaky,
        min_watch_backoff=0.01,
        max_watch_backoff=0.02,
        watch_conn_timeout_sec=0.2,
    )
    await informer.start(timeout=5)
    try:
        async def _await_terminal():
            while True:
                a = await informer.get("a1")
                if a is not None and a.is_terminal():
                    return
                await asyncio.sleep(0.01)

        # Without the fix, call #2 blocks forever, the SUCCEEDED snapshot (call #3)
        # never arrives, and the child stays RUNNING -> this times out.
        await asyncio.wait_for(_await_terminal(), timeout=5)
        assert flaky.calls >= 3
    finally:
        await informer.stop()


@pytest.mark.asyncio
async def test_informer_does_not_recycle_idle_established_stream():
    """A healthy stream that idles (no updates) after the sentinel must NOT be
    torn down by the establishment deadline."""
    run_id = identifier_pb2.RunIdentifier(name="r2", project="p", domain="d")

    class _IdleActions:
        def __init__(self):
            self.calls = 0

        async def watch_for_updates(self, req, **kwargs):
            self.calls += 1
            yield _sentinel()
            await asyncio.sleep(3600)  # established, then idle far longer than the deadline
            yield  # pragma: no cover

    idle = _IdleActions()
    informer = Informer(
        run_id=run_id,
        parent_action_name="a0",
        shared_queue=asyncio.Queue(),
        actions_client=idle,
        min_watch_backoff=0.01,
        max_watch_backoff=0.02,
        watch_conn_timeout_sec=0.05,
    )
    await informer.start(timeout=5)
    try:
        # Far longer than the establishment deadline; a correct impl stays on the
        # single established stream and does not reconnect.
        await asyncio.sleep(0.5)
        assert idle.calls == 1
    finally:
        await informer.stop()
