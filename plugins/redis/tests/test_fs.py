import pathlib

import fakeredis
import flyte.storage as storage
import fsspec
import pytest

from flyteplugins.redis.fs import RedisFileSystem

BASE = "redis://localhost:6379"


@pytest.fixture(autouse=True)
def redis_backend(monkeypatch):
    """Route every RedisFileSystem client to a single in-memory fakeredis server."""
    server = fakeredis.FakeServer()

    def make_client(self, netloc):
        return fakeredis.FakeAsyncRedis(server=server)

    monkeypatch.setattr(RedisFileSystem, "_make_client", make_client)
    # In the plugin's own environment the fsspec.specs entry point registers this class, but
    # register explicitly so the tests also pass from a dev venv without the plugin installed.
    fsspec.register_implementation("redis", RedisFileSystem, clobber=True)
    RedisFileSystem.clear_instance_cache()
    yield server
    RedisFileSystem.clear_instance_cache()


@pytest.mark.asyncio
async def test_pipe_cat_roundtrip():
    fs = fsspec.filesystem("redis")
    await fs._pipe_file(f"{BASE}/a/b/c.pb", b"hello redis")
    assert await fs._cat_file(f"{BASE}/a/b/c.pb") == b"hello redis"


@pytest.mark.asyncio
async def test_cat_missing_raises():
    fs = fsspec.filesystem("redis")
    with pytest.raises(FileNotFoundError):
        await fs._cat_file(f"{BASE}/nope")


@pytest.mark.asyncio
async def test_ranged_read():
    fs = fsspec.filesystem("redis")
    await fs._pipe_file(f"{BASE}/r.bin", b"0123456789")
    assert await fs._cat_file(f"{BASE}/r.bin", start=2, end=5) == b"234"
    assert await fs._cat_file(f"{BASE}/r.bin", start=8, end=100) == b"89"
    assert await fs._cat_file(f"{BASE}/r.bin", start=5, end=5) == b""
    assert await fs._cat_file(f"{BASE}/r.bin", start=-3, end=None) == b"789"


@pytest.mark.asyncio
async def test_info_and_exists_file_and_directory():
    fs = fsspec.filesystem("redis")
    await fs._pipe_file(f"{BASE}/runs/r1/inputs.pb", b"xyz")

    info = await fs._info(f"{BASE}/runs/r1/inputs.pb")
    assert info["type"] == "file"
    assert info["size"] == 3

    info = await fs._info(f"{BASE}/runs/r1")
    assert info["type"] == "directory"

    assert await fs._exists(f"{BASE}/runs/r1/inputs.pb")
    assert await fs._exists(f"{BASE}/runs")
    assert not await fs._exists(f"{BASE}/runs/r2")


@pytest.mark.asyncio
async def test_ls():
    fs = fsspec.filesystem("redis")
    await fs._pipe_file(f"{BASE}/d/one.pb", b"1")
    await fs._pipe_file(f"{BASE}/d/two.pb", b"22")
    await fs._pipe_file(f"{BASE}/d/sub/three.pb", b"333")

    names = await fs._ls(f"{BASE}/d", detail=False)
    assert names == [f"{BASE}/d/one.pb", f"{BASE}/d/sub", f"{BASE}/d/two.pb"]

    details = {i["name"]: i for i in await fs._ls(f"{BASE}/d", detail=True)}
    assert details[f"{BASE}/d/sub"]["type"] == "directory"
    assert details[f"{BASE}/d/two.pb"] == {"name": f"{BASE}/d/two.pb", "size": 2, "type": "file"}


@pytest.mark.asyncio
async def test_rm_file():
    fs = fsspec.filesystem("redis")
    await fs._pipe_file(f"{BASE}/gone.pb", b"x")
    await fs._rm_file(f"{BASE}/gone.pb")
    assert not await fs._exists(f"{BASE}/gone.pb")
    with pytest.raises(FileNotFoundError):
        await fs._rm_file(f"{BASE}/gone.pb")


@pytest.mark.asyncio
async def test_put_stream_get_stream_via_flyte_storage():
    """The exact path flyte metadata IO takes (io.py uses put_stream/get_stream)."""
    payload = b"\x08\x96\x01serialized-literalmap"
    to_path = f"{BASE}/flyte/runs/r1/a0/0/inputs.pb"

    result = await storage.put_stream(payload, to_path=to_path)
    assert result == to_path
    assert await storage.exists(to_path)

    data = b"".join([chunk async for chunk in storage.get_stream(path=to_path)])
    assert data == payload


@pytest.mark.asyncio
async def test_put_stream_empty_payload():
    """A task with no inputs serializes to b'' — the key must still be created."""
    to_path = f"{BASE}/flyte/runs/r2/a0/0/inputs.pb"
    await storage.put_stream(b"", to_path=to_path)
    assert await storage.exists(to_path)
    assert b"".join([chunk async for chunk in storage.get_stream(path=to_path)]) == b""


@pytest.mark.asyncio
async def test_streamed_write_multiple_chunks():
    fs = fsspec.filesystem("redis")
    path = f"{BASE}/big.bin"
    f = await fs.open_async(path, "wb", block_size=8)
    for chunk in (b"aaaa", b"bbbb", b"cccc", b"dd"):
        await f.write(chunk)
    await f.close()
    assert await fs._cat_file(path) == b"aaaabbbbccccdd"

    # wb overwrites
    f = await fs.open_async(path, "wb")
    await f.write(b"fresh")
    await f.close()
    assert await fs._cat_file(path) == b"fresh"


@pytest.mark.asyncio
async def test_storage_get_to_local(tmp_path: pathlib.Path):
    remote = f"{BASE}/dl/file.pb"
    await storage.put_stream(b"download me", to_path=remote)

    local = tmp_path / "file.pb"
    await storage.get(remote, str(local))
    assert local.read_bytes() == b"download me"


@pytest.mark.asyncio
async def test_storage_put_from_local(tmp_path: pathlib.Path):
    local = tmp_path / "up.pb"
    local.write_bytes(b"upload me")

    remote = f"{BASE}/ul/up.pb"
    await storage.put(str(local), remote)
    fs = fsspec.filesystem("redis")
    assert await fs._cat_file(remote) == b"upload me"


@pytest.mark.asyncio
async def test_storage_recursive_get(tmp_path: pathlib.Path):
    await storage.put_stream(b"one", to_path=f"{BASE}/tree/a.pb")
    await storage.put_stream(b"two", to_path=f"{BASE}/tree/sub/b.pb")

    dest = tmp_path / "tree"
    await storage.get(f"{BASE}/tree", str(dest), recursive=True)
    assert (dest / "a.pb").read_bytes() == b"one"
    assert (dest / "sub" / "b.pb").read_bytes() == b"two"


def test_is_remote():
    assert storage.is_remote(f"{BASE}/x")


def test_split_url():
    netloc, key = RedisFileSystem._split_url("redis://user:pw@host:1234/a/b/c.pb")
    assert netloc == "user:pw@host:1234"
    assert key == "a/b/c.pb"
    with pytest.raises(ValueError):
        RedisFileSystem._split_url("s3://bucket/key")
