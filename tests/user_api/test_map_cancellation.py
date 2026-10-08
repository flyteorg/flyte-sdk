"""Tests that flyte.map aborts in-flight child tasks on failure, early exit, and caller cancellation."""

import asyncio
from unittest.mock import Mock, patch

import pytest

import flyte
from flyte._map import MapAsyncIterator, _map


class _TrackingTask:
    """
    Stand-in for AsyncFunctionTaskTemplate. Mirrors the remote controller, which aborts the child action in its
    `CancelledError` handler with an `await`, so cleanup only happens if the cancelled task is allowed to finish.
    """

    def __init__(self, fail_on: int | None = None):
        self.name = "tracking_task"
        self.fail_on = fail_on
        self.started: list[int] = []
        self.aborted: list[int] = []

    async def aio(self, x):
        self.started.append(x)
        if x == self.fail_on:
            raise ValueError(f"boom {x}")
        if x == 0:
            return x
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)  # emulates the `cancel_action` RPC
            self.aborted.append(x)
            raise
        return x


async def _wait_for_started(task: _TrackingTask, n: int):
    for _ in range(500):
        if len(task.started) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"only {len(task.started)} of {n} child tasks started")


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, 3])
async def test_failure_aborts_in_flight_children(concurrency):
    fake = _TrackingTask(fail_on=1)
    it = MapAsyncIterator(func=fake, args=([0, 1, 2, 3],), name="t", concurrency=concurrency, return_exceptions=False)

    with pytest.raises(ValueError, match="boom 1"):
        await it.collect()

    assert sorted(fake.aborted) == sorted(set(fake.started) - {0, 1})
    assert fake.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, 3])
async def test_early_exit_aborts_in_flight_children(concurrency):
    fake = _TrackingTask()
    gen = _map.aio(fake, [0, 1, 2, 3], name="t", concurrency=concurrency)

    async for x in gen:
        assert x == 0
        break
    await gen.aclose()

    assert sorted(fake.aborted) == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, 3])
async def test_public_map_aio_caller_cancellation_aborts_children(concurrency):
    fake = _TrackingTask()
    consumed: list[int] = []

    async def consume():
        async for x in flyte.map.aio(fake, [0, 1, 2, 3], concurrency=concurrency):
            consumed.append(x)

    with patch("flyte.ctx", return_value=Mock(mode="remote")):
        consumer = asyncio.create_task(consume())
        await _wait_for_started(fake, 4 if concurrency == 0 else 3)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    assert consumed == [0]
    assert sorted(fake.aborted) == sorted(set(fake.started) - {0})
    assert fake.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, 3])
async def test_public_map_aio_early_exit_aborts_children(concurrency):
    fake = _TrackingTask()

    with patch("flyte.ctx", return_value=Mock(mode="remote")):
        gen = flyte.map.aio(fake, [0, 1, 2, 3], concurrency=concurrency)
        async for x in gen:
            assert x == 0
            break
        await gen.aclose()

    assert sorted(fake.aborted) == sorted(set(fake.started) - {0})
    assert fake.aborted


@pytest.mark.parametrize("concurrency", [0, 3])
def test_public_map_sync_early_exit_aborts_children(concurrency):
    fake = _TrackingTask()

    with patch("flyte.ctx", return_value=Mock(mode="remote")):
        gen = flyte.map(fake, [0, 1, 2, 3], concurrency=concurrency)
        assert next(gen) == 0
        gen.close()

    assert sorted(fake.aborted) == sorted(set(fake.started) - {0})
    assert fake.aborted
