import asyncio
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.storage


async def main():
    location = "s3://union-cloud-dogfood-1-dogfood/metadata/v2/dogfood/flytesnacks/development/r6tfhn5wxpp44xsw6qmv/a0/1/l6/r6tfhn5wxpp44xsw6qmv-a0-0/711845b49a5b161703f8a8d904b77dc7"
    local_dst = Path("/root/python_download")
    if local_dst.exists():
        local_dst.unlink()

    # time how long it takes to download the file
    start = time.time()
    result = await flyte.storage.get(location, to_path=local_dst)
    print(result)
    end = time.time()
    print(f"Time taken to download the file: {end - start} seconds", flush=True)

    # stream the file
    buf = bytearray(5 * 1024 * 1024 * 1024)
    start = time.time()
    offset = 0
    async for chunk in flyte.storage.get_stream(location, chunk_size=1 * 1024 * 1024):
        end = offset + len(chunk)
        if end > len(buf):
            raise ValueError("Generator produced more data than buffer size")
        buf[offset:end] = chunk
        offset = end

    end = time.time()
    print(f"Time taken to stream file to memory: {end - start} seconds", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
