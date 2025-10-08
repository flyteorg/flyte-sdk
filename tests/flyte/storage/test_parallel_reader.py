import asyncio
from obstore.store import S3Store
from pathlib import Path

import time

from flyte.storage._parallel_reader import ObstoreParallelReader


def test_obstore_parallel_reader_dogfood():
    dogfood_location_prefix = "metadata/v2/dogfood/flytesnacks/development/r6tfhn5wxpp44xsw6qmv/a0/1/l6/r6tfhn5wxpp44xsw6qmv-a0-0/"
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
            Path(location_prefix),
            "/Users/ytong/temp/",
            path_5gb,
        )
    )

    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")


def test_obstore_parallel_reader_sandbox_100_bytes():
    location_prefix = "rand/"
    remote_file = "one_hundred_bytes"
    local_dst = Path("/Users/ytong/temp/downloads/one_hundred_bytes")
    if local_dst.exists():
        local_dst.unlink()


    store = S3Store(
        "bucket",
        endpoint="http://localhost:4566",
        region="us-east-2",
        virtual_hosted_style_request=False,  # path-style works best on many S3-compatibles
        allow_http=True,
        access_key_id="test123", secret_access_key="minio",
        # client_options=ClientConfig(allow_http=True),
    )

    reader = ObstoreParallelReader(store)
    start = time.time()
    asyncio.run(
        reader.download_files(
            Path(location_prefix),
            "/Users/ytong/temp/downloads/",
            remote_file,
        )
    )

    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")
