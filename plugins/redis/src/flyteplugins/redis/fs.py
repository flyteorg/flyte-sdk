"""
fsspec filesystem backed by Redis, for ``redis://`` paths.

This lets flyte store run metadata (inputs.pb / outputs.pb / error.pb) in Redis instead of an
object store, purely path-based: any flyte.storage operation on a ``redis://host:port/some/key``
path is routed here by fsspec. Registration happens lazily through the ``fsspec.specs`` entry
point declared in this plugin's pyproject.toml, so nothing is imported (and redis is not required)
until a ``redis://`` path is actually used.

Path model
----------
``redis://[user:password@]host[:port]/key/path``

The netloc identifies the Redis server (db 0); everything after it is the key, verbatim
(e.g. ``flyte/runs/r1/a0/0/inputs.pb``). Each "file" is a single Redis string value.
Directories are emulated as key prefixes — they exist iff at least one key lives under them.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem
from fsspec.utils import stringify_path

__all__ = ["RedisFileSystem", "RedisStreamedFile"]


def _write_local_file(lpath, data: bytes) -> None:
    parent = os.path.dirname(lpath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(lpath, "wb") as f:
        f.write(data)


def _read_local_file(lpath) -> bytes | None:
    if os.path.isdir(lpath):
        return None
    with open(lpath, "rb") as f:
        return f.read()


class RedisFileSystem(AsyncFileSystem):
    """
    Maps fsspec file semantics onto Redis string keys. Values are bound by Redis's 512 MiB string
    limit, which is far beyond any flyte metadata document; raw data should stay in an object store.
    """

    protocol = "redis"

    def __init__(self, client_factory=None, **storage_options):
        """
        :param client_factory: Optional callable ``(netloc: str) -> redis.asyncio.Redis``,
            primarily for testing (e.g. fakeredis). Defaults to ``redis.asyncio.from_url``.
        """
        super().__init__(**storage_options)
        self._client_factory = client_factory
        # Redis asyncio clients bind connections to the event loop they are first used on. fsspec
        # may drive this filesystem both from the caller's loop (await fs._info(...)) and from its
        # own dedicated IO-loop thread (sync mirrors), so cache one client per (netloc, loop).
        self._clients: dict[tuple[str, int], Any] = {}

    # Paths stay fully qualified (like HTTPFileSystem) because fsspec constructs this filesystem
    # without the path, so the server address must travel inside each path.
    @classmethod
    def _strip_protocol(cls, path):
        return stringify_path(path).rstrip("/")

    def unstrip_protocol(self, name: str) -> str:
        return name

    @staticmethod
    def _split_url(path: str) -> tuple[str, str]:
        """Split a redis URL into (netloc, key)."""
        parsed = urlparse(stringify_path(path))
        if parsed.scheme and parsed.scheme != "redis":
            raise ValueError(f"Not a redis path: {path}")
        return parsed.netloc, parsed.path.lstrip("/")

    def _make_client(self, netloc: str):
        if self._client_factory is not None:
            return self._client_factory(netloc)
        import redis.asyncio as aioredis

        return aioredis.Redis.from_url(f"redis://{netloc}")

    def _client(self, netloc: str):
        key = (netloc, id(asyncio.get_running_loop()))
        if key not in self._clients:
            self._clients[key] = self._make_client(netloc)
        return self._clients[key]

    def _resolve(self, path: str):
        """Return (client, key) for a redis path."""
        netloc, key = self._split_url(path)
        return self._client(netloc), key

    async def _pipe_file(self, path, value, **kwargs):
        client, key = self._resolve(path)
        await client.set(key, bytes(value))

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        client, key = self._resolve(path)
        if start is None and end is None:
            data = await client.get(key)
            if data is None:
                raise FileNotFoundError(path)
            return data

        if not await client.exists(key):
            raise FileNotFoundError(path)
        size = await client.strlen(key)
        start = 0 if start is None else (start + size if start < 0 else start)
        end = size if end is None else (end + size if end < 0 else min(end, size))
        if end <= start:
            return b""
        # GETRANGE is end-inclusive, fsspec ranges are end-exclusive
        return await client.getrange(key, start, end - 1)

    async def _info(self, path, **kwargs):
        client, key = self._resolve(path)
        path = self._strip_protocol(path)
        if key and await client.exists(key):
            return {"name": path, "size": await client.strlen(key), "type": "file"}
        pattern = f"{key}/*" if key else "*"
        async for _ in client.scan_iter(match=pattern, count=100):
            return {"name": path, "size": 0, "type": "directory"}
        raise FileNotFoundError(path)

    async def _ls(self, path, detail=True, **kwargs):
        client, key = self._resolve(path)
        netloc, _ = self._split_url(path)
        prefix = f"{key}/" if key else ""
        base = f"redis://{netloc}/"
        infos: dict[str, dict] = {}

        if key and await client.exists(key):
            name = base + key
            infos[name] = {"name": name, "size": await client.strlen(key), "type": "file"}

        async for raw in client.scan_iter(match=f"{prefix}*", count=100):
            child_key = raw.decode() if isinstance(raw, bytes) else raw
            remainder = child_key[len(prefix) :]
            child = remainder.split("/", 1)[0]
            name = base + prefix + child
            if "/" in remainder:
                infos.setdefault(name, {"name": name, "size": 0, "type": "directory"})
            else:
                infos[name] = {"name": name, "size": await client.strlen(child_key), "type": "file"}

        if not infos:
            raise FileNotFoundError(path)
        out = sorted(infos.values(), key=lambda i: i["name"])
        return out if detail else [i["name"] for i in out]

    async def _rm_file(self, path, **kwargs):
        client, key = self._resolve(path)
        if not await client.delete(key):
            raise FileNotFoundError(path)

    async def _get_file(self, rpath, lpath, **kwargs):
        client, key = self._resolve(rpath)
        data = await client.get(key)
        if data is None:
            if await self._isdir(rpath):
                await asyncio.to_thread(os.makedirs, lpath, exist_ok=True)
                return
            raise FileNotFoundError(rpath)
        await asyncio.to_thread(_write_local_file, lpath, data)

    async def _put_file(self, lpath, rpath, **kwargs):
        data = await asyncio.to_thread(_read_local_file, lpath)
        if data is None:  # directory placeholder; nothing to store
            return
        await self._pipe_file(rpath, data)

    async def open_async(self, path, mode="rb", **kwargs):
        if "b" not in mode:
            raise ValueError("Only binary modes are supported")
        if mode == "rb":
            info = await self._info(path)
            if info["type"] != "file":
                raise IsADirectoryError(path)
            return RedisStreamedFile(self, path, mode, size=info["size"], **kwargs)
        if mode in ("wb", "ab"):
            return RedisStreamedFile(self, path, mode, **kwargs)
        raise NotImplementedError(f"Unsupported mode: {mode}")


class RedisStreamedFile(AbstractAsyncStreamedFile):
    """
    Streamed reads via GETRANGE; streamed writes accumulate with APPEND so a value larger than one
    flush buffer never needs to be held in memory at once.
    """

    async def _fetch_range(self, start, end):
        return await self.fs._cat_file(self.path, start=start, end=end)

    async def _initiate_upload(self):
        if self.mode == "wb":
            client, key = self.fs._resolve(self.path)
            await client.delete(key)

    async def _upload_chunk(self, final=False):
        client, key = self.fs._resolve(self.path)
        data = self.buffer.getvalue()
        if data:
            await client.append(key, data)
        elif final and self.offset == 0:
            # zero-byte file (e.g. an empty serialized LiteralMap) must still create the key
            await client.set(key, b"")
        return True
