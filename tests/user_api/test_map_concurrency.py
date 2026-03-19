"""Tests that flyte.map respects the concurrency parameter."""

import asyncio

import pytest

from flyte._map import MapAsyncIterator


class _FakeTask:
    """Minimal stand-in for AsyncFunctionTaskTemplate that tracks concurrency."""

    def __init__(self, sleep: float = 0.05):
        self.name = "fake_task"
        self._sleep = sleep
        self.peak_concurrent = 0
        self._running = 0
        self._lock = asyncio.Lock()

    async def aio(self, x):
        async with self._lock:
            self._running += 1
            if self._running > self.peak_concurrent:
                self.peak_concurrent = self._running
        await asyncio.sleep(self._sleep)
        async with self._lock:
            self._running -= 1
        return x * 2


@pytest.mark.asyncio
async def test_concurrency_is_respected():
    """With concurrency=3 and 10 tasks, at most 3 should run at once."""
    fake = _FakeTask()
    it = MapAsyncIterator(func=fake, args=(list(range(10)),), name="test", concurrency=3, return_exceptions=True)

    results = await it.collect()
    assert len(results) == 10
    assert results == [i * 2 for i in range(10)]
    assert fake.peak_concurrent <= 3


@pytest.mark.asyncio
async def test_concurrency_zero_means_unlimited():
    """With concurrency=0, all tasks run concurrently."""
    fake = _FakeTask(sleep=0.1)
    it = MapAsyncIterator(func=fake, args=(list(range(10)),), name="test", concurrency=0, return_exceptions=True)

    results = await it.collect()
    assert len(results) == 10
    # All 10 should have been running at the same time
    assert fake.peak_concurrent == 10


@pytest.mark.asyncio
async def test_concurrency_one_is_sequential():
    """With concurrency=1, tasks run one at a time."""
    fake = _FakeTask()
    it = MapAsyncIterator(func=fake, args=(list(range(5)),), name="test", concurrency=1, return_exceptions=True)

    results = await it.collect()
    assert len(results) == 5
    assert results == [i * 2 for i in range(5)]
    assert fake.peak_concurrent == 1


@pytest.mark.asyncio
async def test_concurrency_with_exceptions():
    """Exceptions are returned correctly under concurrency limits."""

    class _FailingTask:
        name = "failing"

        async def aio(self, x):
            if x == 3:
                raise ValueError("boom")
            await asyncio.sleep(0.01)
            return x

    fake = _FailingTask()
    it = MapAsyncIterator(func=fake, args=(list(range(6)),), name="test", concurrency=2, return_exceptions=True)

    results = await it.collect()
    assert len(results) == 6
    assert isinstance(results[3], ValueError)
    for i, r in enumerate(results):
        if i != 3:
            assert r == i


@pytest.mark.asyncio
async def test_concurrency_greater_than_task_count():
    """When concurrency > number of tasks, all tasks run concurrently."""
    fake = _FakeTask(sleep=0.1)
    it = MapAsyncIterator(func=fake, args=(list(range(3)),), name="test", concurrency=100, return_exceptions=True)

    results = await it.collect()
    assert len(results) == 3
    assert fake.peak_concurrent == 3


@pytest.mark.asyncio
async def test_bounded_memory_for_large_input():
    """With concurrency=5 and 50_000 items, peak tasks in-flight stays bounded.

    This simulates the 10M-item scenario at smaller scale: the worker pool
    must NOT materialise all 50k asyncio.Task objects at once.
    """

    class _CountingTask:
        name = "counter"

        def __init__(self):
            self.peak = 0
            self._running = 0
            self._lock = asyncio.Lock()

        async def aio(self, x):
            async with self._lock:
                self._running += 1
                if self._running > self.peak:
                    self.peak = self._running
            # Yield control so other tasks can start
            await asyncio.sleep(0)
            async with self._lock:
                self._running -= 1
            return x

    fake = _CountingTask()
    n = 50_000
    it = MapAsyncIterator(func=fake, args=(list(range(n)),), name="test", concurrency=5, return_exceptions=True)

    results = await it.collect()
    assert len(results) == n
    assert results[0] == 0
    assert results[-1] == n - 1
    # At most concurrency tasks should have been running simultaneously
    assert fake.peak <= 5


@pytest.mark.asyncio
async def test_exception_cancels_remaining_bounded():
    """With return_exceptions=False, an exception cancels remaining work."""

    call_count = 0

    class _Counting:
        name = "counting"

        async def aio(self, x):
            nonlocal call_count
            call_count += 1
            if x == 2:
                raise RuntimeError("stop")
            await asyncio.sleep(0.05)
            return x

    fake = _Counting()
    it = MapAsyncIterator(func=fake, args=(list(range(100)),), name="test", concurrency=2, return_exceptions=False)

    with pytest.raises(RuntimeError, match="stop"):
        await it.collect()

    # With concurrency=2, very few tasks beyond the failing one should have started
    assert call_count < 20
