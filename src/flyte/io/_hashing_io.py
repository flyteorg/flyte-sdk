from __future__ import annotations

import hashlib
import inspect
from typing import Any, Iterable, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class HashMethod(Protocol):
    def update(self, data: memoryview, /) -> None: ...
    def result(self) -> str: ...

    # Optional convenience; not required by the writers.
    def reset(self) -> None: ...


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
