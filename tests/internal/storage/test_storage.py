import os
import tempfile
import unittest

import fsspec
import pytest

import flyte
import flyte.storage as storage
from flyte.storage._storage import (
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
