import os
import tempfile

import flyte
from flyte.io import Dir, File

# To control how offloaded types like Files and Dirs behave with respect to caching, please see the
# content_based_caching.py example.

env = flyte.TaskEnvironment(
    name="dir",
    reusable=flyte.ReusePolicy(replicas=1, concurrency=10),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse"),
)


async def create_test_local_directory() -> str:
    """
    Create a local directory with some test files for demonstration.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="flyte_dir_example_")

    # Create some test files
    with open(os.path.join(temp_dir, "file1.txt"), "w") as f:  # noqa: ASYNC230
        f.write("Content of file 1")

    with open(os.path.join(temp_dir, "file2.txt"), "w") as f:  # noqa: ASYNC230
        f.write("Content of file 2")

    # Create a subdirectory with a file
    sub_dir = os.path.join(temp_dir, "subdir")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "file3.txt"), "w") as f:  # noqa: ASYNC230
        f.write("Content of file 3 in subdirectory")

    print(f"Created test directory at: {temp_dir}")
    return temp_dir


@env.task
async def upload_directory_from_local() -> Dir:
    """
    Demonstrates Dir.from_local() - uploading a local directory asynchronously.
    """
    local_path = await create_test_local_directory()
    uploaded_dir = await Dir.from_local(local_path)
    print(f"Uploaded local directory {local_path} to remote: {uploaded_dir.path}")
    return uploaded_dir


@env.task
async def create_reference_to_existing_remote(remote_path: str) -> Dir:
    """
    Demonstrates Dir.from_existing_remote() - referencing an existing remote directory.
    """
    dir_ref = Dir.from_existing_remote(remote_path)
    print(f"Created Dir reference to existing remote directory: {dir_ref.path}")
    return dir_ref


@env.task
async def check_directory_exists(d: Dir) -> bool:
    """
    Demonstrates Dir.exists() - checking if a directory exists asynchronously.
    """
    exists = await d.exists()
    print(f"Directory {d.path} exists: {exists}")
    return exists


@env.task
async def list_files_in_directory(d: Dir) -> list[File]:
    """
    Demonstrates Dir.list_files() - getting a list of files in the directory (non-recursive).
    """
    files = await d.list_files()
    print(f"Found {len(files)} files in directory {d.path}:")
    for file in files:
        print(f"  - {file.name}: {file.path}")
    return files


@env.task
async def walk_directory_async(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk() - asynchronously walking through the directory.
    """
    all_files = []
    print(f"Walking directory {d.path} (recursive):")
    async for file in d.walk(recursive=True):
        print(f"  Found file: {file.name} at {file.path}")
        all_files.append(file)
    return all_files


@env.task
async def walk_directory_non_recursive(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk() - walking directory non-recursively.
    """
    files = []
    print(f"Walking directory {d.path} (non-recursive):")
    async for file in d.walk(recursive=False):
        print(f"  Found file: {file.name} at {file.path}")
        files.append(file)
    return files


@env.task
async def walk_directory_with_max_depth(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk() - walking directory with max depth limit.
    """
    files = []
    print(f"Walking directory {d.path} (max depth 2):")
    async for file in d.walk(recursive=True, max_depth=2):
        print(f"  Found file: {file.name} at {file.path}")
        files.append(file)
    return files


@env.task
async def get_specific_file(d: Dir, file_name: str) -> File | None:
    """
    Demonstrates Dir.get_file() - getting a specific file from the directory.
    """
    file = await d.get_file(file_name)
    if file:
        print(f"Found file {file_name}: {file.path}")
        return file
    else:
        print(f"File {file_name} not found in directory {d.path}")
        return None


@env.task
async def read_files_in_directory(d: Dir) -> dict[str, str]:
    """
    Demonstrates reading the contents of files in a directory.
    """
    file_contents = {}
    async for file in d.walk(recursive=False):
        if file.name.endswith(".txt"):  # Only read text files
            try:
                async with file.open("rb") as f:
                    content = bytes(await f.read()).decode("utf-8")
                    file_contents[file.name] = content
                    print(f"Read {file.name}: {content}")
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                file_contents[file.name] = f"Error: {e}"

    return file_contents


@env.task
async def download_directory_async(d: Dir) -> str:
    """
    Demonstrates Dir.download() - downloading a directory asynchronously.
    """
    local_path = await d.download()
    print(f"Downloaded directory to: {local_path}")

    # List what was downloaded
    print("Downloaded files:")
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, local_path)
            print(f"  - {rel_path}")

    return local_path


@env.task
async def download_directory_to_specific_path(d: Dir) -> str:
    """
    Demonstrates Dir.download() - downloading to a specific local path.
    """
    temp_dir = tempfile.mkdtemp(prefix="flyte_downloaded_dir_")
    local_path = await d.download(temp_dir)
    print(f"Downloaded directory to specific path: {local_path}")
    return local_path


@env.task
async def demonstrate_directory_properties(d: Dir) -> None:
    """
    Demonstrates accessing Dir properties.
    """
    print(f"Directory path: {d.path}")
    print(f"Directory name: {d.name}")
    print(f"Directory format: {d.format}")
    print(f"Directory hash: {d.hash}")


@env.task
async def create_directory_with_cache_key() -> Dir:
    """
    Demonstrates creating a Dir with a specific cache key.
    """
    # Create a local directory first
    temp_dir = tempfile.mkdtemp(prefix="flyte_cache_key_example_")
    with open(os.path.join(temp_dir, "cached_file.txt"), "w") as f:  # noqa: ASYNC230
        f.write("This directory has a specific cache key")

    # Upload with a cache key
    dir_with_cache = await Dir.from_local(temp_dir, dir_cache_key="my_custom_cache_key_123")
    print(f"Created directory with cache key: {dir_with_cache.hash}")
    return dir_with_cache


@env.task
async def process_files_in_parallel(d: Dir) -> dict[str, int]:
    """
    Demonstrates processing multiple files in a directory concurrently.
    """
    import asyncio

    async def process_file(file: File) -> tuple[str, int]:
        """Process a single file and return its name and size."""
        try:
            async with file.open("rb") as f:
                content = await f.read()
                size = len(content)
                print(f"Processed {file.name}: {size} bytes")
                return file.name, size
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            return file.name, 0

    # Get all files
    files = []
    async for file in d.walk(recursive=True):
        files.append(file)

    # Process all files concurrently
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    return dict(results)


@env.task
async def main() -> None:
    """
    Main function demonstrating all Dir async APIs.
    """
    print("=== Flyte Dir Async API Examples ===\n")

    # 2. Upload directory from local
    print("\n2. Uploading directory from local...")
    remote_dir = await upload_directory_from_local()

    # 3. Check if directory exists
    print("\n3. Checking if directory exists...")
    await check_directory_exists(remote_dir)

    # 4. List files in directory (non-recursive)
    print("\n4. Listing files in directory (non-recursive)...")
    files = await list_files_in_directory(remote_dir)
    print(f"\nTotal files found (non-recursive): {len(files)}")

    # 5. Walk directory recursively
    print("\n5. Walking directory recursively...")
    all_files = await walk_directory_async(remote_dir)
    print("\nTotal files found recursively:", len(all_files))

    # 6. Walk directory non-recursively
    print("\n6. Walking directory non-recursively...")
    await walk_directory_non_recursive(remote_dir)

    # 7. Walk directory with max depth
    print("\n7. Walking directory with max depth...")
    await walk_directory_with_max_depth(remote_dir)

    # 8. Get specific file
    print("\n8. Getting specific file...")
    specific_file = await get_specific_file(remote_dir, "file1.txt")
    if specific_file:
        print(f"Specific file path: {specific_file.path}")
    else:
        print("Specific file not found.")

    # 9. Read file contents
    print("\n9. Reading file contents...")
    file_contents = await read_files_in_directory(remote_dir)
    print(f"File contents: {file_contents}")

    # 10. Download directory
    print("\n10. Downloading directory...")
    downloaded_path = await download_directory_async(remote_dir)
    print(f"Directory downloaded to: {downloaded_path}")

    # 11. Download to specific path
    print("\n11. Downloading to specific path...")
    await download_directory_to_specific_path(remote_dir)

    # 12. Create reference to existing remote
    print("\n12. Creating reference to existing remote...")
    dir_ref = await create_reference_to_existing_remote(remote_dir.path)
    print(f"Referenced directory path: {dir_ref.path}")

    # 13. Demonstrate directory properties
    print("\n13. Directory properties...")
    await demonstrate_directory_properties(remote_dir)

    # 14. Create directory with cache key
    print("\n14. Creating directory with cache key...")
    cached_dir = await create_directory_with_cache_key()
    print(f"Directory with cache key path: {cached_dir.path}, hash: {cached_dir.hash}")

    # 15. Process files in parallel
    print("\n15. Processing files in parallel...")
    file_sizes = await process_files_in_parallel(remote_dir)
    print(f"File sizes: {file_sizes}")

    print("\n=== All Dir async API examples completed! ===")


if __name__ == "__main__":

    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
