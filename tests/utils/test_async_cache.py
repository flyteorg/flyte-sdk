import asyncio
import threading

import pytest

from flyte._utils import AsyncLRUCache
from flyte._utils.async_cache import loop_agnostic_async_cache


@pytest.mark.asyncio
async def test_async_lru_cache_basic():
    cache = AsyncLRUCache[str, int](maxsize=3)

    # Test with sync value function
    counter = 0

    def compute_value():
        nonlocal counter
        counter += 1
        return counter

    # First access computes the value
    assert await cache.get("key1", compute_value) == 1
    # Second access uses cached value
    assert await cache.get("key1", compute_value) == 1
    assert counter == 1

    # Different key computes new value
    assert await cache.get("key2", compute_value) == 2
    assert counter == 2

    # LRU eviction - key1 is still in cache, adding key3 should evict key2
    assert await cache.get("key3", compute_value) == 3
    assert counter == 3

    # key1 should still be cached
    assert await cache.get("key1", compute_value) == 1
    assert counter == 3

    assert await cache.get("key4", compute_value) == 4
    assert counter == 4

    # key2 should have been evicted
    assert await cache.get("key2", compute_value) == 5
    assert counter == 5


@pytest.mark.asyncio
async def test_async_value_function():
    cache = AsyncLRUCache[str, str](maxsize=10)

    async def async_compute():
        await asyncio.sleep(0.1)
        return "async_result"

    result = await cache.get("async_key", async_compute)
    assert result == "async_result"

    # Should use cached value
    result = await cache.get("async_key", async_compute)
    assert result == "async_result"


@pytest.mark.asyncio
async def test_ttl_expiration():
    cache = AsyncLRUCache[str, int](maxsize=10, ttl=0.2)

    counter = 0

    def compute_value():
        nonlocal counter
        counter += 1
        return counter

    # First access
    assert await cache.get("key", compute_value) == 1
    # Before expiration
    assert await cache.get("key", compute_value) == 1

    # Wait for TTL to expire
    await asyncio.sleep(0.3)

    # After expiration, should compute again
    assert await cache.get("key", compute_value) == 2


@pytest.mark.asyncio
async def test_concurrent_access():
    cache = AsyncLRUCache[str, int](maxsize=10)

    counter = 0
    delay = 0.2

    async def slow_compute():
        nonlocal counter
        await asyncio.sleep(delay)
        counter += 1
        return counter

    # Launch multiple concurrent requests for the same key
    tasks = [cache.get("concurrent_key", slow_compute) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # All results should be the same and counter should be 1
    assert all(r == 1 for r in results)
    assert counter == 1


@pytest.mark.asyncio
async def test_direct_set_and_contains():
    cache = AsyncLRUCache[str, int](maxsize=10)

    # Set a value directly
    await cache.set("direct_key", 42)

    # Check contains
    assert await cache.contains("direct_key")
    assert not await cache.contains("missing_key")

    # Get should return the directly set value
    value = await cache.get("direct_key", lambda: 99)
    assert value == 42


@pytest.mark.asyncio
async def test_invalidate():
    cache = AsyncLRUCache[str, int](maxsize=10)

    counter = 0

    def compute_value():
        nonlocal counter
        counter += 1
        return counter

    # First access
    assert await cache.get("key", compute_value) == 1

    # Invalidate
    await cache.invalidate("key")

    # Should compute again
    assert await cache.get("key", compute_value) == 2


class _Service:
    """Stand-in for a cluster-aware service: one instance, many event loops."""

    def __init__(self):
        self.calls = 0

    @loop_agnostic_async_cache()
    async def resolve(self, key: str) -> str:
        self.calls += 1
        await asyncio.sleep(0)
        return f"{key}-client-{self.calls}"

    @loop_agnostic_async_cache(maxsize=2)
    async def small(self, key: str) -> int:
        self.calls += 1
        return self.calls

    @loop_agnostic_async_cache()
    async def flaky(self, key: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return "ok"

    @loop_agnostic_async_cache()
    async def slow(self, key: str) -> int:
        self.calls += 1
        await asyncio.sleep(0.2)
        return self.calls


@pytest.mark.asyncio
async def test_loop_agnostic_cache_memoizes():
    svc = _Service()
    assert await svc.resolve("a") == "a-client-1"
    assert await svc.resolve("a") == "a-client-1"
    assert await svc.resolve("b") == "b-client-2"
    assert svc.calls == 2


def test_loop_agnostic_cache_survives_loop_change(recwarn):
    """The regression this decorator exists for: alru_cache would clear the cache and warn."""
    svc = _Service()

    assert asyncio.run(svc.resolve("a")) == "a-client-1"
    # A second loop (e.g. the tracked-run reporter thread) reuses the cached value.
    assert asyncio.run(svc.resolve("a")) == "a-client-1"
    assert svc.calls == 1
    assert [w for w in recwarn if "event loop change" in str(w.message)] == []


def test_loop_agnostic_cache_shared_across_threads():
    svc = _Service()
    results = []

    def run_in_thread():
        results.append(asyncio.run(svc.resolve("a")))

    threads = [threading.Thread(target=run_in_thread) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Racing threads may each resolve once, but every later call reuses a single value.
    assert len(set(results)) <= len(threads)
    assert asyncio.run(svc.resolve("a")) in results


@pytest.mark.asyncio
async def test_loop_agnostic_cache_dedupes_concurrent_callers():
    svc = _Service()
    results = await asyncio.gather(*[svc.slow("a") for _ in range(5)])
    assert results == [1, 1, 1, 1, 1]
    assert svc.calls == 1


@pytest.mark.asyncio
async def test_loop_agnostic_cache_does_not_cache_failures():
    svc = _Service()
    with pytest.raises(RuntimeError, match="boom"):
        await svc.flaky("a")
    assert await svc.flaky("a") == "ok"
    assert await svc.flaky("a") == "ok"
    assert svc.calls == 2


@pytest.mark.asyncio
async def test_loop_agnostic_cache_evicts_least_recently_used():
    svc = _Service()
    assert await svc.small("a") == 1
    assert await svc.small("b") == 2
    assert await svc.small("a") == 1  # 'a' is now the most recently used
    assert await svc.small("c") == 3  # evicts 'b'
    assert await svc.small("a") == 1
    assert await svc.small("b") == 4


@pytest.mark.asyncio
async def test_loop_agnostic_cache_is_per_instance():
    first, second = _Service(), _Service()
    assert await first.resolve("a") == "a-client-1"
    assert await second.resolve("a") == "a-client-1"
    assert first.calls == 1 and second.calls == 1


@pytest.mark.asyncio
async def test_loop_agnostic_cache_caller_cancellation_does_not_cancel_shared_call():
    svc = _Service()
    task = asyncio.create_task(svc.slow("a"))
    await asyncio.sleep(0.01)
    other = asyncio.create_task(svc.slow("a"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await other == 1
    assert svc.calls == 1
