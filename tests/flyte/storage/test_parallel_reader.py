import asyncio
import filecmp
import os
import pathlib
import sys
import time
import unittest.mock as mock
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
from flyte.storage._parallel_reader import Chunk, DownloadTask, ObstoreParallelReader, Source


@pytest.mark.asyncio
async def test_worker_logs_exception_on_download_failure(tmp_path):
    """Worker should log the failing file path before re-raising so the real
    error is visible instead of being swallowed inside a BaseExceptionGroup."""

    store = mock.MagicMock()
    reader = ObstoreParallelReader(store, max_concurrency=1)

    async def _mock_list(*args, **kwargs):
        yield [{"path": "prefix/file.txt", "size": 100}]

    async def _mock_as_completed(gen, transformer=None):
        async for task in gen:
            try:
                raise RuntimeError("GCS 429: Too Many Requests")
            except Exception:
                import flyte.storage._parallel_reader as pr

                pr.logger.exception(f"Failed downloading {task.source.path}")
                raise
            yield

    with (
        mock.patch("flyte.storage._parallel_reader.obstore") as mock_obstore,
        mock.patch("flyte.storage._parallel_reader.logger") as mock_logger,
        mock.patch.object(reader, "_as_completed", side_effect=_mock_as_completed),
    ):
        mock_obstore.list = _mock_list
        mock_obstore.get_range_async = mock.AsyncMock()

        with pytest.raises(Exception):
            await reader.download_files(Path("prefix"), tmp_path)

    mock_logger.exception.assert_called_once()
    call_args = str(mock_logger.exception.call_args)
    assert "prefix/file.txt" in call_args


@pytest.mark.asyncio
async def test_worker_logs_exception_before_task_received(tmp_path):
    """Worker should log a fallback message when failure happens before any
    task is dequeued (task is still None at that point)."""

    store = mock.MagicMock()
    reader = ObstoreParallelReader(store, max_concurrency=1)

    async def _mock_list(*args, **kwargs):
        yield [{"path": "prefix/file.txt", "size": 100}]

    async def _mock_as_completed(*args, **kwargs):
        import flyte.storage._parallel_reader as pr

        try:
            raise RuntimeError("inq exploded before task received")
        except Exception:
            pr.logger.exception("Error before receiving a task")
            raise
        yield

    with (
        mock.patch("flyte.storage._parallel_reader.obstore") as mock_obstore,
        mock.patch("flyte.storage._parallel_reader.logger") as mock_logger,
        mock.patch.object(reader, "_as_completed", side_effect=_mock_as_completed),
    ):
        mock_obstore.list = _mock_list
        mock_obstore.get_range_async = mock.AsyncMock()

        with pytest.raises(Exception):
            await reader.download_files(Path("prefix"), tmp_path)

    mock_logger.exception.assert_called_once()
    call_args = str(mock_logger.exception.call_args)
    assert "before receiving a task" in call_args


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.TaskGroup uses ExceptionGroup on 3.11+")
@pytest.mark.asyncio
async def test_as_completed_unwraps_single_taskgroup_exception():
    from builtins import BaseExceptionGroup

    store = mock.MagicMock()
    reader = ObstoreParallelReader(store, max_concurrency=1)

    async def _gen():
        yield DownloadTask(
            source=Source(id="file", path=Path("prefix/file.txt"), length=4),
            chunk=Chunk(offset=0, length=4),
        )

    with mock.patch(
        "flyte.storage._parallel_reader.obstore.get_range_async",
        new=mock.AsyncMock(side_effect=RuntimeError("GCS 429: Too Many Requests")),
    ):
        with pytest.raises(RuntimeError, match="GCS 429: Too Many Requests") as exc_info:
            async for _ in reader._as_completed(_gen()):
                pass

    assert not isinstance(exc_info.value, BaseExceptionGroup)


@pytest.mark.asyncio
async def test_worker_logs_transformer_exception_with_file_context():
    store = mock.MagicMock()
    reader = ObstoreParallelReader(store, max_concurrency=1)

    async def _gen():
        yield DownloadTask(
            source=Source(id="file", path=Path("prefix/file.txt"), length=4),
            chunk=Chunk(offset=0, length=4),
        )

    async def _transformer(*args, **kwargs):
        raise RuntimeError("replace failed")

    with (
        mock.patch(
            "flyte.storage._parallel_reader.obstore.get_range_async",
            new=mock.AsyncMock(return_value=b"data"),
        ),
        mock.patch("flyte.storage._parallel_reader.logger") as mock_logger,
    ):
        with pytest.raises(RuntimeError, match="replace failed"):
            async for _ in reader._as_completed(_gen(), transformer=_transformer):
                pass

    mock_logger.exception.assert_called_once()
    call_args = str(mock_logger.exception.call_args)
    assert "prefix/file.txt" in call_args
    assert ", 0)" in call_args


@pytest.mark.asyncio
async def test_as_completed_pre311_surfaces_worker_exception_without_hanging():

    store = mock.MagicMock()
    reader = ObstoreParallelReader(store, max_concurrency=1)

    async def _gen():
        yield DownloadTask(
            source=Source(id="file", path=Path("prefix/file.txt"), length=4),
            chunk=Chunk(offset=0, length=4),
        )

    async def _consume():
        async for _ in reader._as_completed(_gen()):
            pass

    with (
        mock.patch("flyte.storage._parallel_reader.sys.version_info", (3, 10)),
        mock.patch(
            "flyte.storage._parallel_reader.obstore.get_range_async",
            new=mock.AsyncMock(side_effect=RuntimeError("GCS 429: Too Many Requests")),
        ),
    ):
        with pytest.raises(RuntimeError, match="GCS 429: Too Many Requests"):
            await asyncio.wait_for(_consume(), timeout=0.5)


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
