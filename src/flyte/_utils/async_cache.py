import asyncio
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, Generic, Optional, TypeVar

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
                value = await value_func()
            else:
                value = value_func()  # type: ignore

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


# Example usage:
"""

"""
