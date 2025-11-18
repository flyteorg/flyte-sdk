import os
import tempfile

import aiofiles

import flyte
from flyte.io import File

# To control how offloaded types like Files and Dirs behave with respect to caching, please see the
# content_based_caching.py example.

env = flyte.TaskEnvironment(
    name="file",
    reusable=flyte.ReusePolicy(replicas=1, concurrency=10),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)


@env.task
async def create_new_remote_file(content: str) -> File:
    """
    Demonstrates File.new_remote() - creating a new remote file asynchronously.
    """
    f = File.new_remote()
    async with f.open("wb") as fh:
        await fh.write(content.encode("utf-8"))
    print(f"Created new remote file at: {f.path}")
    return f


@env.task
async def read_file_async(f: File) -> str:
    """
    Demonstrates reading a file asynchronously using open().
    """
    async with f.open("rb") as fh:
        contents = bytes(await fh.read())  # read() returns a memoryview, NOTE the bytes() conversion
        text_content = contents.decode("utf-8")
        print(f"File {f.path} contents: {text_content}")
        return text_content


@env.task
async def download_file_async(f: File) -> str:
    """
    Demonstrates File.download() - downloading a file to local storage asynchronously.
    """
    local_path = await f.download()
    print(f"Downloaded file to: {local_path}")

    # Verify the download worked
    async with aiofiles.open(local_path, "rb") as fh:
        contents = await fh.read()
        text_content = contents.decode("utf-8")
        print(f"Downloaded file contents: {text_content}")

    return local_path


@env.task
async def upload_local_file_async(content: str) -> File:
    """
    Demonstrates File.from_local() - uploading a local file asynchronously.
    """
    # Create a temporary local file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Upload the local file
        uploaded_file = await File.from_local(tmp_path)
        print(f"Uploaded local file {tmp_path} to remote: {uploaded_file.path}")
        return uploaded_file
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


@env.task
async def create_from_existing_remote(remote_path: str) -> File:
    """
    Demonstrates File.from_existing_remote() - referencing an existing remote file.
    """
    f = File.from_existing_remote(remote_path)
    print(f"Created File reference to existing remote file: {f.path}")
    return f


@env.task
async def check_file_exists_sync(f: File) -> bool:
    """
    Demonstrates File.exists_sync() - checking if a file exists (sync method).
    Note: There's no async version of exists, so we use the sync version.
    """
    exists = f.exists_sync()
    print(f"File {f.path} exists: {exists}")
    return exists


@env.task
async def copy_file_by_reference(f: File) -> File:
    """
    Demonstrates creating a File by direct reference (no upload/download).
    """
    # This creates a new File object pointing to the same remote location
    copied_file = File(path=f.path)
    print(f"Created file reference to: {copied_file.path}")
    return copied_file


@env.task
async def demonstrate_file_properties(f: File) -> None:
    """
    Demonstrates accessing File properties.
    """
    print(f"File path: {f.path}")
    print(f"File name: {f.name}")
    print(f"File format: {f.format}")
    print(f"File hash: {f.hash}")


@env.task
async def demonstrate_streaming_write(content: str) -> File:
    """
    Demonstrates streaming write to a remote file.
    """
    f = File.new_remote()
    async with f.open("wb") as fh:
        # Write in chunks to demonstrate streaming
        data = content.encode("utf-8")
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            await fh.write(chunk)
    print(f"Streamed content to file: {f.path}")
    return f


@env.task
async def demonstrate_streaming_read(f: File) -> str:
    """
    Demonstrates streaming read from a remote file.
    """
    print(f"File path: {f.path}", flush=True)
    content_parts = []
    async with f.open("rb", block_size=10) as fh:
        # Read in chunks to demonstrate streaming
        while True:
            chunk = await fh.read()
            if not chunk:
                break
            content_parts.append(chunk)

    full_content = b"".join(content_parts).decode("utf-8")
    print(f"Streamed content from file: {full_content}")
    return full_content


@env.task
async def main() -> None:
    """
    Main function demonstrating all File async APIs.
    """
    print("=== Flyte File Async API Examples ===\n")

    # 1. Create a new remote file
    print("1. Creating new remote file...")
    file1 = await create_new_remote_file("Hello, Flyte Async API!")

    # 2. Read the file
    print("\n2. Reading file contents...")
    await read_file_async(file1)

    # 3. Check if file exists
    print("\n3. Checking if file exists...")
    await check_file_exists_sync(file1)

    # 4. Download the file
    print("\n4. Downloading file...")
    await download_file_async(file1)

    # 5. Upload a local file
    print("\n5. Uploading local file...")
    file2 = await upload_local_file_async("Content from local file!")

    # 6. Create reference to existing file
    print("\n6. Creating reference to existing file...")
    file3 = await create_from_existing_remote(file1.path)
    print("\nReferenced file path:", file3.path)

    # 7. Copy file by reference
    print("\n7. Copying file by reference...")
    await copy_file_by_reference(file2)

    # 8. Demonstrate file properties
    print("\n8. File properties...")
    await demonstrate_file_properties(file1)

    # 10. Demonstrate streaming operations
    print("\n10. Streaming write...")
    stream_file = await demonstrate_streaming_write("This is streaming content that will be written in chunks!")

    print("\n11. Streaming read...")
    await demonstrate_streaming_read(stream_file)

    print("\n=== All File async API examples completed! ===")


if __name__ == "__main__":

    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
