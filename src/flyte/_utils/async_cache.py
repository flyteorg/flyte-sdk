import asyncio
import functools
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Coroutine, Dict, Generic, Hashable, Optional, Tuple, TypeVar, cast

from typing_extensions import ParamSpec

K = TypeVar("K")
V = TypeVar("V")


class AsyncLRUCache(Generic[K, V]):
    """
    A high-performance async-compatible LRU cache.

    Examples:
    ```python
    # Create a cache instance
    cache = AsyncLRUCache[str, dict](maxsize=100)

    async def fetch_data(user_id: str) -> dict:
        # Define the expensive operation as a local function
        async def get_user_data():
            await asyncio.sleep(1)  # Simulating network/DB delay
            return {"id": user_id, "name": f"User {user_id}"}

    # Use the cache
    return await cache.get(f"user:{user_id}", get_user_data)
    ```
    This cache can be used from async coroutines and handles concurrent access safely.
    """

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Initialize the async LRU cache.

        Args:
            maxsize: Maximum number of items to keep in the cache
            ttl: Time-to-live for cache entries in seconds, or None for no expiration
        """
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._locks: Dict[K, asyncio.Lock] = {}
        self._access_lock = asyncio.Lock()

    async def get(self, key: K, value_func: Callable[[], V | Awaitable[V]]) -> V:
        """
        Get a value from the cache, computing it if necessary.

        Args:
            key: The cache key
            value_func: Function or coroutine to compute the value if not cached

        Returns:
            The cached or computed value
        """
        # Fast path: check if key exists and is not expired
        if key in self._cache:
            value, timestamp = self._cache[key]
            if self._ttl is None or time.time() - timestamp < self._ttl:
                # Move the accessed item to the end (most recently used)
                async with self._access_lock:
                    self._cache.move_to_end(key)
                return value

        # Slow path: compute the value
        # Get or create a lock for this key to prevent redundant computation
        async with self._access_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock

        async with lock:
            # Check again in case another coroutine computed the value while we waited
            if key in self._cache:
                value, timestamp = self._cache[key]
                if self._ttl is None or time.time() - timestamp < self._ttl:
                    async with self._access_lock:
                        self._cache.move_to_end(key)
                    return value

            # Compute the value
            if asyncio.iscoroutinefunction(value_func):
                value = cast(V, await value_func())
            else:
                value = cast(V, value_func())

            # Store in cache
            async with self._access_lock:
                self._cache[key] = (value, time.time())
                # Evict least recently used items if needed
                while len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)
                # Clean up the lock
                self._locks.pop(key, None)

            return value

    async def set(self, key: K, value: V) -> None:
        """
        Explicitly set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        async with self._access_lock:
            self._cache[key] = (value, time.time())
            # Evict least recently used items if needed
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    async def invalidate(self, key: K) -> None:
        """Remove a specific key from the cache."""
        async with self._access_lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._access_lock:
            self._cache.clear()
            self._locks.clear()

    async def contains(self, key: K) -> bool:
        """Check if a key exists in the cache and is not expired."""
        if key not in self._cache:
            return False

        if self._ttl is None:
            return True

        _, timestamp = self._cache[key]
        return time.time() - timestamp < self._ttl


P = ParamSpec("P")
R = TypeVar("R")

# Guards the per-instance cache dictionaries below. Only ever held around a few dict
# operations — never across an await — so a single process-wide lock is cheap and keeps
# instances usable from several threads/event loops at once.
_cache_state_lock = threading.Lock()


def _make_key(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Hashable:
    if not kwargs:
        return args
    return args, tuple(sorted(kwargs.items()))


def _finish(
    results: "OrderedDict[Hashable, Any]",
    in_flight: Dict[Hashable, "asyncio.Task[Any]"],
    key: Hashable,
    maxsize: int,
    task: "asyncio.Task[Any]",
) -> None:
    """Done-callback: retire the in-flight task and cache its value if it succeeded."""
    with _cache_state_lock:
        if in_flight.get(key) is task:
            del in_flight[key]
        # Calling exception() also marks it retrieved, so asyncio won't log a failure
        # whose only waiter was cancelled. Failures are deliberately not cached.
        if task.cancelled() or task.exception() is not None:
            return
        results[key] = task.result()
        while len(results) > maxsize:
            results.popitem(last=False)


def loop_agnostic_async_cache(
    maxsize: int = 128,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """
    Memoize an async method's result so the cache survives a change of event loop.

    `alru_cache` caches the `asyncio.Task` it created for each key, which binds every entry
    to the loop that produced it. async_lru >= 2.3.0 detects when the same cache is used
    from a second loop, clears it, and emits `AlruCacheLoopResetWarning`. Flyte drives one
    set of clients from several loops in a single process — the CLI's `asyncio.run` loop,
    the syncify background loop, the controller thread, the tracked-run reporter thread —
    so that reset fires during ordinary runs, printing a warning and discarding entries
    that were perfectly reusable (the values cached here are pyqwest-backed clients, which
    hold no event-loop state).

    This decorator caches the resolved value instead of the task, in per-instance state, so
    every loop shares it. Concurrent callers on the same loop still collapse onto a single
    in-flight call, and failures are never cached.

    The decorated method must belong to a class whose instances accept attribute
    assignment (no `__slots__`), and its arguments must be hashable.

    Examples:
    ```python
    class Service:
        @loop_agnostic_async_cache()
        async def _resolve(self, org: str) -> Client:
            return await self._build_client(org)
    ```

    :param maxsize: Maximum number of cached results to keep per instance, least recently
        used evicted first.
    """

    def decorator(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        name = getattr(fn, "__name__", f"fn{id(fn)}")
        results_attr = f"_flyte_cached_results_{name}"
        in_flight_attr = f"_flyte_in_flight_{name}"

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> R:
            key = _make_key(args, kwargs)
            loop = asyncio.get_running_loop()
            with _cache_state_lock:
                results: Optional[OrderedDict[Hashable, R]] = getattr(self, results_attr, None)
                if results is None:
                    results = OrderedDict()
                    setattr(self, results_attr, results)
                if key in results:
                    results.move_to_end(key)
                    return results[key]

                # In-flight tasks are the one loop-bound piece of state, so they are tracked
                # per loop and dropped with it.
                by_loop: Optional[
                    weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[Hashable, "asyncio.Task[R]"]]
                ] = getattr(self, in_flight_attr, None)
                if by_loop is None:
                    by_loop = weakref.WeakKeyDictionary()
                    setattr(self, in_flight_attr, by_loop)
                in_flight = by_loop.get(loop)
                if in_flight is None:
                    in_flight = {}
                    by_loop[loop] = in_flight

                task = in_flight.get(key)
                if task is None:
                    task = loop.create_task(fn(self, *args, **kwargs))  # ty: ignore[missing-argument]
                    in_flight[key] = task
                    task.add_done_callback(functools.partial(_finish, results, in_flight, key, maxsize))

            # Shielded so that one caller giving up doesn't cancel the call the others share.
            return await asyncio.shield(task)

        return cast(Callable[P, Coroutine[Any, Any, R]], wrapper)

    return decorator
