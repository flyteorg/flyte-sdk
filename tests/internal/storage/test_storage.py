import os
import tempfile
import unittest

import pytest

import flyte
import flyte.storage as storage
from flyte.storage._storage import (
    _get_stream_obstore_bypass,
    _is_obstore_supported_protocol,
    _put_stream_obstore_bypass,
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


@pytest.mark.asyncio
async def test_obstore_bypass():
    """
    Test that the obstore bypass is working correctly.
    """
    data = "Hello, world!".encode("utf-8")
    dst = os.path.join(tempfile.mkdtemp(), "dst")
    await _put_stream_obstore_bypass(data, to_path=dst)
    streams = _get_stream_obstore_bypass(dst, chunk_size=10 * 1024 * 1024)  # 10 MB chunk size
    assert data == b"".join([chunk async for chunk in streams])


@pytest.mark.asyncio
async def test_obstore_bypass_with_large_data():
    """
    Test that the obstore bypass works with large data.
    """
    data = os.urandom(10 * 1024 * 1024)
    dst = os.path.join(tempfile.mkdtemp(), "large_dst")
    await _put_stream_obstore_bypass(data, to_path=dst)
    streams = _get_stream_obstore_bypass(dst, chunk_size=10 * 1024)
    assert data == b"".join([chunk async for chunk in streams])


@pytest.mark.asyncio
async def test_obstore_bypass_with_empty_data():
    """
    Test that the obstore bypass works with empty data.
    """
    data = b""
    dst = os.path.join(tempfile.mkdtemp(), "empty_dst")
    await _put_stream_obstore_bypass(data, to_path=dst)
    streams = _get_stream_obstore_bypass(dst, chunk_size=10 * 1024)
    assert data == b"".join([chunk async for chunk in streams])
