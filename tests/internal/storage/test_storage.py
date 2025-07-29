import os
import tempfile
import unittest

import pytest

import flyte
import flyte.storage as storage


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
