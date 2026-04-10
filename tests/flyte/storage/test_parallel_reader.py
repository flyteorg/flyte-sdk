import asyncio
import filecmp
import os
import pathlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import aiofiles.os
import pytest
from obstore.store import S3Store

import flyte
from flyte import storage
from flyte._context import internal_ctx
from flyte.storage import S3
from flyte.storage._parallel_reader import ObstoreParallelReader


@pytest.mark.skip
@pytest.mark.asyncio
async def test_access_large_file():
    location = "s3://bucket/metadata/v2/testorg/testproject/development/rxw4wk5fdw9tfl24pnv9/a0/1/f3/rxw4wk5fdw9tfl24pnv9-a0-0/b087922792e194f32f601d1083ef02f5"
    local_dst = Path("/Users/ytong/temp/b087922792e194f32f601d1083ef02f5")
    if local_dst.exists():
        local_dst.unlink()

    s3_cfg = S3.for_sandbox()
    await flyte.init.aio(storage=s3_cfg)

    # time how long it takes to download the file
    start = time.time()
    result = await storage.get(location, to_path=local_dst)
    print(result)
    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds", flush=True)

    # stream the file
    buf = bytearray(5 * 1024 * 1024 * 1024)
    start = time.time()
    offset = 0
    async for chunk in storage.get_stream(location, chunk_size=1 * 1024 * 1024):
        end = offset + len(chunk)
        if end > len(buf):
            raise ValueError("Generator produced more data than buffer size")
        buf[offset:end] = chunk
        offset = end

    end = time.time()
    print(f"Time taken to stream file to memory: {end - start} seconds", flush=True)


@pytest.mark.skip
def test_obstore_parallel_reader_dogfood():
    dogfood_location_prefix = (
        "metadata/v2/dogfood/flytesnacks/development/r6tfhn5wxpp44xsw6qmv/a0/1/l6/r6tfhn5wxpp44xsw6qmv-a0-0/"
    )
    path_5gb = "711845b49a5b161703f8a8d904b77dc7"
    local_dst = Path("/root/model_loader_output")
    if local_dst.exists():
        local_dst.unlink()

    store = S3Store(
        "union-cloud-dogfood-1-dogfood",
        # endpoint="http://localhost:4566",
        region="us-east-2",
        virtual_hosted_style_request=False,  # path-style works best on many S3-compatibles
        # allow_http=True,
        # access_key_id="test123", secret_access_key="minio",
        # client_options=ClientConfig(allow_http=True),
    )

    reader = ObstoreParallelReader(store)
    start = time.time()
    asyncio.run(
        reader.download_files(
            Path(dogfood_location_prefix),
            "/Users/ytong/temp/",
            path_5gb,
        )
    )

    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_obstore_parallel_reader_sandbox_100_bytes(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    await flyte.init.aio(storage=S3.for_sandbox())

    pp = internal_ctx().raw_data.path
    parsed = urlparse(pp)
    bucket = parsed.netloc  # "bucket"
    location_prefix = parsed.path.lstrip("/")
    print(f"Using raw data path: {pp}. Bucket {bucket}, {location_prefix=}", flush=True)

    # create a random 100-byte file in locally and put to localstack

    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))
    remote_location = await storage.put(str(local_file))
    path_for_reader = remote_location.replace(pp, "")
    print(f"Uploaded temp file {local_file} to {remote_location}", flush=True)
    print(f"Attempting to download {path_for_reader=}", flush=True)

    store = S3Store(
        bucket,
        endpoint="http://localhost:4566",
        region="us-east-2",
        virtual_hosted_style_request=False,  # path-style works best on many S3-compatibles
        allow_http=True,
        access_key_id="test123",
        secret_access_key="minio",
        # client_options=ClientConfig(allow_http=True),
    )

    reader = ObstoreParallelReader(store)
    start = time.time()
    await reader.download_files(
        Path(location_prefix),
        tmp_path / "downloaded",
        path_for_reader,
    )

    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")
    assert filecmp.cmp(local_file, tmp_path / "downloaded" / path_for_reader, shallow=False)


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_obstore_parallel_reader_with_storage_100_bytes(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    await flyte.init.aio(storage=S3.for_sandbox())

    pp = internal_ctx().raw_data.path
    parsed = urlparse(pp)
    bucket = parsed.netloc  # "bucket"
    location_prefix = parsed.path.lstrip("/")
    print(f"Using raw data path: {pp}. Bucket {bucket}, {location_prefix=}", flush=True)

    # create a random 100-byte file in locally and put to localstack

    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))
    remote_location = await storage.put(str(local_file))
    path_for_reader = remote_location.replace(pp, "")
    print(f"Uploaded temp file {local_file} to {remote_location}", flush=True)
    print(f"Attempting to download {path_for_reader=}", flush=True)

    start = time.time()
    await storage.get(remote_location, to_path=tmp_path / "downloaded" / path_for_reader)
    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")
    assert filecmp.cmp(local_file, tmp_path / "downloaded" / path_for_reader, shallow=False)


# ---------------------------------------------------------------------------
# Unit tests — no sandbox / real S3 required
# ---------------------------------------------------------------------------


def _make_mock_head(file_path: pathlib.Path, content: bytes):
    """Return an async mock for obstore.head_async that reports a single file."""

    async def _head(store, path):
        return {
            "path": str(file_path),
            "size": len(content),
            "last_modified": None,
            "e_tag": None,
            "version": None,
        }

    return _head


def _make_mock_get_range(content: bytes):
    """Return an async mock for obstore.get_range_async that slices bytes."""

    async def _get_range(store, path, *, start, end):
        return content[start:end]

    return _get_range


@pytest.mark.asyncio
async def test_download_files_single_file(tmp_path):
    """Unit test: download a single file without a real object store."""
    content = b"hello parallel reader"
    src_prefix = pathlib.Path("prefix/v2/org/project/dev/eid")
    file_rel = "bundle.pkl"
    file_path = src_prefix / file_rel

    store = MagicMock()
    with (
        patch(
            "flyte.storage._parallel_reader.obstore.head_async",
            side_effect=_make_mock_head(file_path, content),
        ),
        patch(
            "flyte.storage._parallel_reader.obstore.get_range_async",
            side_effect=_make_mock_get_range(content),
        ),
    ):
        reader = ObstoreParallelReader(store, chunk_size=len(content))
        await reader.download_files(src_prefix, tmp_path / "output", file_rel)

    result = tmp_path / "output" / file_rel
    assert result.exists(), "downloaded file must exist"
    assert result.read_bytes() == content


@pytest.mark.asyncio
async def test_download_files_multipart(tmp_path):
    """Unit test: verify a file split across multiple chunks reassembles correctly."""
    content = os.urandom(256)
    src_prefix = pathlib.Path("prefix/v2/org/project/dev/eid")
    file_rel = "big.bin"
    file_path = src_prefix / file_rel

    store = MagicMock()
    # Use a small chunk size so the file is split into several chunks
    chunk_size = 64
    with (
        patch(
            "flyte.storage._parallel_reader.obstore.head_async",
            side_effect=_make_mock_head(file_path, content),
        ),
        patch(
            "flyte.storage._parallel_reader.obstore.get_range_async",
            side_effect=_make_mock_get_range(content),
        ),
    ):
        reader = ObstoreParallelReader(store, chunk_size=chunk_size)
        await reader.download_files(src_prefix, tmp_path / "output", file_rel)

    result = tmp_path / "output" / file_rel
    assert result.read_bytes() == content


@pytest.mark.asyncio
async def test_download_files_no_cross_device_error(tmp_path):
    """
    Regression test for OSError [Errno 18] Invalid cross-device link (EXDEV).

    In some container images /tmp is on a separate tmpfs mount from the task
    working directory (e.g. /root).  Before this fix, ObstoreParallelReader
    created its staging temp dir in /tmp and then called os.rename() to move
    each file to the target directory — which fails with EXDEV when crossing
    device boundaries.

    The fix passes ``dir=target_prefix`` to TemporaryDirectory so the staging
    temp dir lives on the same filesystem as the destination.

    This test simulates the cross-device scenario by wrapping aiofiles.os.replace
    so it raises EXDEV whenever the *source* path is outside target_prefix.
    With the fix in place the source is always inside target_prefix, so the
    wrapper passes through and the download completes successfully.
    """
    content = b"cross-device regression content"
    src_prefix = pathlib.Path("prefix/v2/org/project/dev/eid")
    file_rel = "fast.tar.gz"
    file_path = src_prefix / file_rel
    target_prefix = tmp_path / "output"

    # Capture the real aiofiles.os.replace before we patch it
    _real_replace = aiofiles.os.replace

    async def _cross_device_replace(src, dst):
        """Raise EXDEV if src is not inside target_prefix (simulates /tmp on separate device)."""
        try:
            pathlib.Path(src).relative_to(target_prefix)
        except ValueError:
            raise OSError(18, "Invalid cross-device link", str(src), None, str(dst))
        return await _real_replace(src, dst)

    store = MagicMock()
    with (
        patch(
            "flyte.storage._parallel_reader.obstore.head_async",
            side_effect=_make_mock_head(file_path, content),
        ),
        patch(
            "flyte.storage._parallel_reader.obstore.get_range_async",
            side_effect=_make_mock_get_range(content),
        ),
        patch("aiofiles.os.replace", side_effect=_cross_device_replace),
    ):
        reader = ObstoreParallelReader(store, chunk_size=len(content))
        # Must not raise OSError [Errno 18]
        await reader.download_files(src_prefix, target_prefix, file_rel)

    result = target_prefix / file_rel
    assert result.exists(), "downloaded file must exist after cross-device-safe move"
    assert result.read_bytes() == content
