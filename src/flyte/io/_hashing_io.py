from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Iterable, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class HashMethod(Protocol):
    def update(self, data: Any, /) -> None: ...
    def result(self) -> str: ...

    # Optional convenience; not required by the writers.
    def reset(self) -> None: ...


class HashFunction(HashMethod):
    """A hash method that wraps a user-provided function to compute hashes.

    This class allows you to define custom hashing logic by providing a callable
    that takes data and returns a hash string. It implements the HashMethod protocol,
    making it compatible with Flyte's hashing infrastructure.

    Example:
        >>> def my_hash(data: bytes) -> str:
        ...     return hashlib.md5(data).hexdigest()
        >>> hash_fn = HashFunction.from_fn(my_hash)
        >>> hash_fn.update(b"hello")
        >>> hash_fn.result()
        '5d41402abc4b2a76b9719d911017c592'

    Attributes:
        _fn: The callable that computes the hash from input data.
        _value: The most recently computed hash value.
    """

    def __init__(self, fn: Callable[[Any], str]):
        """Initialize a HashFunction with a custom hash callable.

        Args:
            fn: A callable that takes data of any type and returns a hash string.
        """
        self._fn = fn

    def update(self, data: Any):
        """Update the hash value by applying the hash function to the given data.

        Args:
            data: The data to hash. The type depends on the hash function provided.
        """
        self._value = self._fn(data)

    def result(self) -> str:
        """Return the most recently computed hash value.

        Returns:
            The hash string from the last call to update().
        """
        return self._value

    @classmethod
    def from_fn(cls, fn: Callable[[Any], str]) -> HashFunction:
        """Create a HashFunction from a callable.

        This is a convenience factory method for creating HashFunction instances.

        Args:
            fn: A callable that takes data of any type and returns a hash string.

        Returns:
            A new HashFunction instance wrapping the provided callable.

        Example:
            >>> hash_fn = HashFunction.from_fn(lambda x: hashlib.sha256(x).hexdigest())
        """
        return cls(fn)


class PrecomputedValue(HashMethod):
    def __init__(self, value: str):
        self._value = value

    def update(self, data: memoryview, /) -> None: ...

    def result(self) -> str:
        return self._value


class HashlibAccumulator(HashMethod):
    """
    Wrap a hashlib-like object to the Accumulator protocol.
    h = hashlib.new("sha256")
    acc = HashlibAccumulator(h)
    """

    def __init__(self, h):
        self._h = h

    def update(self, data: memoryview, /) -> None:
        self._h.update(data)

    def result(self) -> Any:
        return self._h.hexdigest()

    @classmethod
    def from_hash_name(cls, name: str) -> HashlibAccumulator:
        """
        Create an accumulator from a hashlib algorithm name.
        """
        h = hashlib.new(name)
        return cls(h)


class HashingWriter:
    """
    Sync writer that updates a user-supplied accumulator on every write.

    Hashing covers the exact bytes you pass in. If you write str, it is encoded
    using the underlying file's .encoding if available, else UTF-8 (for hashing only).
    """

    def __init__(
        self,
        fh,
        accumulator: HashMethod,
        *,
        encoding: Optional[str] = None,
        errors: str = "strict",
    ):
        self._fh = fh
        self._acc: HashMethod = accumulator
        self._encoding = encoding or getattr(fh, "encoding", None)
        self._errors = errors

    def result(self) -> Any:
        return self._acc.result()

    def _to_bytes_mv(self, data) -> memoryview:
        if isinstance(data, str):
            b = data.encode(self._encoding or "utf-8", self._errors)
            return memoryview(b)
        if isinstance(data, (bytes, bytearray, memoryview)):
            return memoryview(data)
        # Accept any buffer-protocol object (e.g., numpy arrays)
        return memoryview(data)

    def write(self, data):
        mv = self._to_bytes_mv(data)
        self._acc.update(mv)
        return self._fh.write(data)

    def writelines(self, lines: Iterable[Union[str, bytes, bytearray, memoryview]]):
        for line in lines:
            self.write(line)

    def flush(self):
        return self._fh.flush()

    def close(self):
        return self._fh.close()

    def __getattr__(self, name):
        return getattr(self._fh, name)


class AsyncHashingWriter:
    """
    Async version of HashingWriter with the same behavior.
    """

    def __init__(
        self,
        fh,
        accumulator: HashMethod,
        *,
        encoding: Optional[str] = None,
        errors: str = "strict",
    ):
        self._fh = fh
        self._acc = accumulator
        self._encoding = encoding or getattr(fh, "encoding", None)
        self._errors = errors

    def result(self) -> Any:
        return self._acc.result()

    def _to_bytes_mv(self, data) -> memoryview:
        if isinstance(data, str):
            b = data.encode(self._encoding or "utf-8", self._errors)
            return memoryview(b)
        if isinstance(data, (bytes, bytearray, memoryview)):
            return memoryview(data)
        return memoryview(data)

    async def write(self, data):
        mv = self._to_bytes_mv(data)
        self._acc.update(mv)
        return await self._fh.write(data)

    async def writelines(self, lines: Iterable[Union[str, bytes, bytearray, memoryview]]):
        for line in lines:
            await self.write(line)

    async def flush(self):
        fn = getattr(self._fh, "flush", None)
        if fn is None:
            return None
        res = fn()
        if inspect.isawaitable(res):
            return await res
        return res

    async def close(self):
        fn = getattr(self._fh, "close", None)
        if fn is None:
            return None
        res = fn()
        if inspect.isawaitable(res):
            return await res
        return res

    def __getattr__(self, name):
        return getattr(self._fh, name)


class HashingReader:
    """
    Sync reader that updates a user-supplied accumulator on every read operation.

    If the underlying handle returns str (text mode), we encode it for hashing only,
    using the handle's .encoding if present, else the explicit 'encoding' arg, else UTF-8.
    """

    def __init__(
        self,
        fh,
        accumulator: HashMethod,
        *,
        encoding: Optional[str] = None,
        errors: str = "strict",
    ):
        self._fh = fh
        self._acc = accumulator
        self._encoding = encoding or getattr(fh, "encoding", None)
        self._errors = errors

    def result(self) -> str:
        return self._acc.result()

    def _to_bytes_mv(self, data) -> Optional[memoryview]:
        if data is None:
            return None
        if isinstance(data, str):
            return memoryview(data.encode(self._encoding or "utf-8", self._errors))
        if isinstance(data, (bytes, bytearray, memoryview)):
            return memoryview(data)
        # Accept any buffer-protocol object (rare for read paths, but safe)
        return memoryview(data)

    def read(self, size: int = -1):
        data = self._fh.read(size)
        mv = self._to_bytes_mv(data)
        if mv is not None and len(mv) > 0:
            self._acc.update(mv)
        return data

    def readline(self, size: int = -1):
        line = self._fh.readline(size)
        mv = self._to_bytes_mv(line)
        if mv is not None and len(mv) > 0:
            self._acc.update(mv)
        return line

    def readlines(self, hint: int = -1):
        lines = self._fh.readlines(hint)
        # Update in order to reflect exact concatenation
        for line in lines:
            mv = self._to_bytes_mv(line)
            if mv is not None and len(mv) > 0:
                self._acc.update(mv)
        return lines

    def __iter__(self):
        return self

    def __next__(self):
        # Delegate to the underlying iterator to preserve semantics (including buffering),
        # but intercept the produced line to hash it.
        line = next(self._fh)
        mv = self._to_bytes_mv(line)
        if mv is not None and len(mv) > 0:
            self._acc.update(mv)
        return line

    # ---- passthrough ----
    def __getattr__(self, name):
        # Avoid leaking private lookups to the underlying object if we're missing something internal
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._fh, name)


class AsyncHashingReader:
    """
    Async reader that updates a user-supplied accumulator on every read operation.

    Works with aiofiles/fsspec async handles. `flush`/`close` may be awaitable or sync.
    """

    def __init__(
        self,
        fh,
        accumulator: HashMethod,
        *,
        encoding: Optional[str] = None,
        errors: str = "strict",
    ):
        self._fh = fh
        self._acc = accumulator
        self._encoding = encoding or getattr(fh, "encoding", None)
        self._errors = errors

    def result(self) -> str:
        return self._acc.result()

    def _to_bytes_mv(self, data) -> Optional[memoryview]:
        if data is None:
            return None
        if isinstance(data, str):
            return memoryview(data.encode(self._encoding or "utf-8", self._errors))
        if isinstance(data, (bytes, bytearray, memoryview)):
            return memoryview(data)
        return memoryview(data)

    async def read(self, size: int = -1):
        data = await self._fh.read(size)
        mv = self._to_bytes_mv(data)
        if mv is not None and len(mv) > 0:
            self._acc.update(mv)
        return data

    async def readline(self, size: int = -1):
        line = await self._fh.readline(size)
        mv = self._to_bytes_mv(line)
        if mv is not None and len(mv) > 0:
            self._acc.update(mv)
        return line

    async def readlines(self, hint: int = -1):
        # Some async filehandles implement readlines(); if not, fall back to manual loop.
        if hasattr(self._fh, "readlines"):
            lines = await self._fh.readlines(hint)
            for line in lines:
                mv = self._to_bytes_mv(line)
                if mv is not None and len(mv) > 0:
                    self._acc.update(mv)
            return lines
        # Fallback: read all via iteration
        lines = []
        async for line in self:
            lines.append(line)
        return lines

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Prefer the underlying async iterator if present
        anext_fn = getattr(self._fh, "__anext__", None)
        if anext_fn is not None:
            try:
                line = await anext_fn()
            except StopAsyncIteration:
                raise
            mv = self._to_bytes_mv(line)
            if mv is not None and len(mv) > 0:
                self._acc.update(mv)
            return line

        # Fallback to readline-based iteration
        line = await self.readline()
        if line == "" or line == b"":
            raise StopAsyncIteration
        return line

    async def flush(self):
        fn = getattr(self._fh, "flush", None)
        if fn is None:
            return None
        res = fn()
        return await res if inspect.isawaitable(res) else res

    async def close(self):
        fn = getattr(self._fh, "close", None)
        if fn is None:
            return None
        res = fn()
        return await res if inspect.isawaitable(res) else res

    # ---- passthrough ----
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._fh, name)
