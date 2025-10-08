import asyncio
import time
from pathlib import Path

from obstore.store import S3Store
from union._model_loader.loader import ObstoreParallelReader

CHUNK_SIZE = 16 * 1024 * 1024
MAX_CONCURRENCY = 32


def obstore_parallel_reader():
    location_prefix = (
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
            Path(location_prefix),
            "/Users/ytong/temp/",
            path_5gb,
        )
    )

    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds")


if __name__ == "__main__":
    obstore_parallel_reader()

"""
code to test rustfs, ended up being a bit slower than s5cmd
import time
import rustfs
location = "s3://union-cloud-dogfood-1-dogfood/metadata/v2/dogfood/flytesnacks/development/r6tfhn5wxpp44xsw6qmv/a0/1/l6/r6tfhn5wxpp44xsw6qmv-a0-0/711845b49a5b161703f8a8d904b77dc7"
start = time.time()
fs = rustfs.RustFileSystem()
fs.get(location, "/root/downloaded_by_rustfs")
end = time.time()
print(f"Time taken to download the file: {end - start} seconds")
"""
