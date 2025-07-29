import asyncio

import pytest

from flyte._utils import AsyncLRUCache


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
