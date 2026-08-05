import math
import os
import tempfile
import unittest

import fsspec
import pytest

import flyte
import flyte.storage as storage
from flyte.storage._storage import (
    _MAX_SAFE_PARTS,
    _UPLOAD_CHUNK_FLOOR,
    _UPLOAD_MAX_CONCURRENCY,
    _compute_upload_chunk_size,
    _is_obstore_supported_protocol,
    _open_obstore_bypass,
)


class TestUnionFileSystem(unittest.IsolatedAsyncioTestCase):
    flyte.init()

    async def test_file(self):
        temp_file = tempfile.mktemp()
        data = os.urandom(10)
        with open(temp_file, "wb") as f:  # noqa: ASYNC230
            f.write(data)

        source = os.path.join(tempfile.mkdtemp(), "source")
        dst = os.path.join(tempfile.mkdtemp(), "dst")

        await storage.put(temp_file, dst)
        await storage.get(dst, source)

    async def test_stream(self):
        data = os.urandom(10)

        dst = os.path.join(tempfile.mkdtemp(), "dst")
        await storage.put_stream(data, to_path=dst)
        streams = storage.get_stream(dst)
        assert data == b"".join([chunk async for chunk in streams])


@pytest.mark.parametrize(
    "protocol,storage_config_class",
    [
        ("s3", storage.S3),
        ("gs", storage.GCS),
        ("abfs", storage.ABFS),
        ("abfss", storage.ABFS),
    ],
)
def test_known_protocols(protocol, storage_config_class):
    kwargs = storage.get_configured_fsspec_kwargs(protocol=protocol)
    assert kwargs == storage_config_class.auto().get_fsspec_kwargs()


def test_obstore_protocol():
    assert _is_obstore_supported_protocol("s3")
    assert _is_obstore_supported_protocol("gs")
    assert _is_obstore_supported_protocol("abfs")
    assert _is_obstore_supported_protocol("abfss")
    assert not _is_obstore_supported_protocol("obstore")


@pytest.fixture(scope="module")
def obstore_file():
    """
    Fixture to create a temporary obstore file for testing.
    """
    import fsspec.implementations.local
    import obstore.fsspec

    obstore.fsspec.register("file", asynchronous=True)
    try:
        yield
    finally:
        fsspec.register_implementation("file", fsspec.implementations.local.LocalFileSystem, clobber=True)


@pytest.mark.asyncio
async def test_obstore_bypass(obstore_file):
    """
    Test that the obstore bypass is working correctly.
    """
    data = "Hello, world!".encode("utf-8")
    dst = os.path.join(tempfile.mkdtemp(), "dst")
    fh = await _open_obstore_bypass(dst, "wb")
    await fh.write(data)
    await fh.close()
    # Read back the data in chunks
    fh = await _open_obstore_bypass(dst, "rb", chunk_size=5 * 1024)
    rcv_data = await fh.read()
    fh.close()
    assert data == rcv_data


@pytest.mark.asyncio
async def test_obstore_bypass_with_large_data(obstore_file):
    """
    Test that the obstore bypass works with large data.
    """
    chunk_size = 10 * 1024 * 1024
    data = os.urandom(chunk_size)
    dst = os.path.join(tempfile.mkdtemp(), "large_dst")
    fh = await _open_obstore_bypass(dst, "wb", chunk_size=chunk_size)
    await fh.write(data)
    await fh.close()
    # Read back the data in chunks
    rcv_data = []
    fh = await _open_obstore_bypass(dst, "rb", chunk_size=5 * 1024)
    while chunk := await fh.read():
        rcv_data.append(chunk)
    fh.close()
    assert data == b"".join(rcv_data)


@pytest.mark.asyncio
async def test_obstore_bypass_with_empty_data(obstore_file):
    """
    Test that the obstore bypass works with empty data.
    """
    data = b""
    dst = os.path.join(tempfile.mkdtemp(), "empty_dst")
    fh = await _open_obstore_bypass(dst, "wb")
    await fh.write(data)
    await fh.close()
    # Read back the data in chunks
    fh = await _open_obstore_bypass(dst, "rb", chunk_size=5 * 1024)
    rcv_data = await fh.read()
    fh.close()
    assert data == rcv_data


@pytest.mark.asyncio
async def test_storage_exists():
    fsspec.register_implementation("file", fsspec.implementations.local.LocalFileSystem, clobber=True)
    assert await storage.exists("/tmp")
    import os

    listed = os.listdir("/tmp")[0]
    assert await storage.exists(os.path.join("/tmp", listed)), f"{listed} not found"
    assert not await storage.exists("/non-existent/test")


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_get_underlying_filesystem_upload_download(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Sandbox integration test that uses get_underlying_filesystem with the sandbox S3
    (LocalStack) to upload and download a file.
    """
    from flyte.storage import S3

    await flyte.init.aio(storage=S3.for_sandbox())

    # Create a local file with known content
    original_content = b"hello from sandbox integration test"
    local_file = tmp_path / "upload_me.txt"
    local_file.write_bytes(original_content)

    # Upload the file to sandbox S3 via storage.put
    s3_path = "s3://bucket/tests/default_upload/upload_me.txt"
    await storage.put(str(local_file), s3_path)

    # Use get_underlying_filesystem to verify the file exists on S3
    fs = storage.get_underlying_filesystem(path=s3_path)
    assert fs.exists(s3_path)

    # Download the file back using get_underlying_filesystem
    downloaded_file = tmp_path / "downloaded.txt"
    fs.get(s3_path, str(downloaded_file))

    # Verify the downloaded content matches the original
    assert downloaded_file.read_bytes() == original_content

    # Also upload via the filesystem directly and read back with storage.get
    s3_path_2 = "s3://bucket/tests/default_upload/fs_uploaded.txt"
    fs.put(str(local_file), s3_path_2)

    downloaded_file_2 = tmp_path / "downloaded_2.txt"
    await storage.get(s3_path_2, str(downloaded_file_2))
    assert downloaded_file_2.exists()


def test_compute_upload_chunk_size():
    # Small / unknown files keep the 5 MiB floor.
    assert _compute_upload_chunk_size(None) == _UPLOAD_CHUNK_FLOOR
    assert _compute_upload_chunk_size(0) == _UPLOAD_CHUNK_FLOOR
    assert _compute_upload_chunk_size(1024 * 1024) == _UPLOAD_CHUNK_FLOOR
    # A file right at the 5 MiB * 9000 boundary still uses the floor.
    assert _compute_upload_chunk_size(_UPLOAD_CHUNK_FLOOR * _MAX_SAFE_PARTS) == _UPLOAD_CHUNK_FLOOR

    # Large files scale the part size up so the part count stays under the 10,000 hard limit.
    big = 60 * 2**30  # 60 GiB -- well past the ~48.8 GiB ceiling of a fixed 5 MiB part size
    cs = _compute_upload_chunk_size(big)
    assert cs > _UPLOAD_CHUNK_FLOOR
    assert math.ceil(big / cs) <= _MAX_SAFE_PARTS
    assert math.ceil(big / cs) <= 10000


class _FakeStore:
    def __init__(self, captured):
        self._captured = captured

    async def put_async(self, remote_key, local, *, chunk_size, max_concurrency):
        self._captured.append(
            {
                "remote_key": remote_key,
                "local": str(local),
                "chunk_size": chunk_size,
                "max_concurrency": max_concurrency,
            }
        )


class _FakeObstoreFS:
    """Minimal stand-in for an obstore-backed AsyncFileSystem to exercise the put bypass."""

    protocol = "gs"

    def __init__(self, captured):
        self._captured = captured

    def _split_path(self, path):
        p = path.replace("gs://", "")
        bucket, _, key = p.partition("/")
        return bucket, key

    def _construct_store(self, bucket):
        return _FakeStore(self._captured)


@pytest.mark.asyncio
async def test_put_routes_through_obstore_bypass(monkeypatch):
    from flyte.storage import _storage

    captured = []
    monkeypatch.setattr(_storage, "get_underlying_filesystem", lambda path: _FakeObstoreFS(captured))

    temp_file = tempfile.mktemp()
    with open(temp_file, "wb") as f:  # noqa: ASYNC230
        f.write(os.urandom(1024))

    result = await storage.put(temp_file, "gs://bucket/path/to/obj")

    assert result == "gs://bucket/path/to/obj"
    assert len(captured) == 1
    assert captured[0]["remote_key"] == "path/to/obj"
    assert captured[0]["chunk_size"] == _UPLOAD_CHUNK_FLOOR
    assert captured[0]["max_concurrency"] == _UPLOAD_MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_put_obstore_bypass_recursive(monkeypatch):
    from flyte.storage import _storage

    captured = []
    monkeypatch.setattr(_storage, "get_underlying_filesystem", lambda path: _FakeObstoreFS(captured))

    src = tempfile.mkdtemp()
    os.makedirs(os.path.join(src, "sub"))
    for rel in ["a.txt", os.path.join("sub", "b.txt")]:
        with open(os.path.join(src, rel), "wb") as f:  # noqa: ASYNC230
            f.write(b"x")

    await storage.put(src, "gs://bucket/prefix", recursive=True)

    keys = sorted(c["remote_key"] for c in captured)
    assert keys == ["prefix/a.txt", "prefix/sub/b.txt"]
    assert all(c["max_concurrency"] == _UPLOAD_MAX_CONCURRENCY for c in captured)


@pytest.mark.parametrize(
    "path,expected",
    [
        # Linux paths - not remote
        ("/path/to/file", False),
        ("/home/user/data.txt", False),
        ("./relative/path", False),
        ("relative/path/file.txt", False),
        # Windows paths - not remote
        ("C:\\Users\\test\\file.txt", False),
        ("D:\\data\\folder", False),
        ("C:/Users/test/file.txt", False),
        # file:// protocol - not remote
        ("file:///path/to/file", False),
        ("file:///home/user/data.txt", False),
        ("file://C:/Users/test/file.txt", False),
        # S3 paths - remote
        ("s3://bucket/path/to/file", True),
        ("s3://my-bucket/data.parquet", True),
        # GCS paths - remote
        ("gs://bucket/path/to/file", True),
        ("gs://my-bucket/data.parquet", True),
        # Azure paths - remote
        ("abfs://container/path/to/file", True),
        ("abfss://container/path/to/file", True),
        ("abfs://my-container@storageaccount.dfs.core.windows.net/data", True),
    ],
)
def test_is_remote(path, expected):
    assert storage.is_remote(path) == expected
