import asyncio
import filecmp
import os
import time
from pathlib import Path
from urllib.parse import urlparse

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
