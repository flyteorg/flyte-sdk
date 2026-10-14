import asyncio
import threading
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, Generic, Hashable, Optional, TypeVar, cast

K = TypeVar("K", bound=Hashable)
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


class ThreadSafeCache(Generic[K, V]):
    """A tiny threadsafe memo shared across event loops.

    It memoizes an async factory's resolved *value* (never its `asyncio.Task`), so one
    instance can safely serve every event loop and thread in the process. Flyte drives a
    single set of cluster-aware clients from several loops — the CLI's `asyncio.run` loop,
    the syncify background loop, a controller thread, the tracked-run reporter thread — and
    the cached clients hold no event-loop state, so sharing them is safe. A task-based cache
    such as async_lru's `alru_cache` instead binds each entry to the loop that produced it;
    on a loop change it clears itself and emits `AlruCacheLoopResetWarning`, discarding
    perfectly reusable values. Keeping the resolved value (not the task) avoids that.

    The mapping is guarded by a short `threading.Lock`. Lookups happen under it, but the
    async factory cannot run under a lock, so it runs outside on a miss and only successful
    results are stored — a failed resolution is never cached and the next call retries.
    Concurrent misses for the same key each resolve independently (SelectCluster is
    idempotent) rather than being serialized.

    Examples:
    ```python
    cache = ThreadSafeCache[str, Client]()

    async def resolve(org: str) -> Client:
        def build() -> Awaitable[Client]:
            return self._build_client(org)
        return await cache.get_or_compute(org, build)
    ```
    """

    def __init__(self, maxsize: int = 128):
        self._cache: "OrderedDict[K, V]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    async def get_or_compute(self, key: K, factory: Callable[[], Awaitable[V]]) -> V:
        """Return the cached value for `key`, computing it with `factory` on a miss.

        Args:
            key: Hashable lookup key for the value.
            factory: Async callable producing the value; invoked only on a cache miss and
                only its successful result is stored.

        Returns:
            The cached or freshly computed value.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        value = await factory()

        with self._lock:
            self._cache[key] = value
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
            return value
