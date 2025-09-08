"""
Normally when using caching, Flyte computes a cache key based on the Cache version and the inputs to the task.
The inputs are hashed using some sane default, but in some cases you may want to override the hash of an input.
The simplest way to do this is to just provide a pre-computed constant value for the hash.

Keep in mind caching during local execution in general has not been implemented yet, so this example will fail
locally. If you're interested in contributing support, please let us know!
"""

import tempfile
from pathlib import Path

import flyte
from flyte import Cache
from flyte.io import Dir, File

env = flyte.TaskEnvironment(name="content-based-caching")


@env.task(cache=Cache(behavior="override", version_override="v4"))
async def process_file_with_preset_hash(input_file: File) -> str:
    """
    This task processes a file. This task should not run if the set hash for the File is unchanged.
    """
    import random

    print(f"Processing file: {input_file.path}")
    print(f"File hash: {input_file.hash}")

    random_num = random.randint(1, 1000000)
    async with input_file.open("rb") as f:
        # when running locally, fsspec filesystem is not async.
        content = f.read()
        content = content.decode("utf-8")
        lines = content.strip().split("\n")
        return f"Processed {len(lines)} lines from {input_file.name} - Random: {random_num}"


@env.task(cache=Cache(behavior="override", version_override="v4"))
async def process_directory_with_preset_hash(input_dir: Dir) -> str:
    """
    This task processes a directory. Should not run if the set hash for the Dir is unchanged.
    """
    import random

    print(f"Processing directory: {input_dir.path}")
    print(f"Directory hash: {input_dir.hash}")

    random_num = random.randint(1, 1000000)
    files = await input_dir.list_files()
    total_size = 0

    for file in files:
        # For demonstration, just count the files
        total_size += 1

    return f"Processed directory with {total_size} files - Random: {random_num}"


@env.task
async def demo_cache_behavior() -> str:
    """
    Demonstrate how content-based caching works with preset hashes.
    Different file contents with same hash should hit the cache.
    """
    # Create first file with some content and preset hash
    content1 = "Hello, Flyte!\nThis is file version 1"
    hash_value = "test_hash_12345"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(content1)
        temp_path1 = temp_file.name

    file1 = await File.from_local(temp_path1, hash_method=hash_value)

    # Create first directory with preset hash
    with tempfile.TemporaryDirectory() as temp_dir1:
        (Path(temp_dir1) / "file1.txt").write_text("Content of file 1")
        (Path(temp_dir1) / "file2.txt").write_text("Content of file 2")
        dir1 = await Dir.from_local(temp_dir1, dir_cache_key="dir_hash_67890")

    # Process both - these should run normally (first time)
    print("=== First run (should execute tasks) ===")
    result1_file = await process_file_with_preset_hash(file1)
    result1_dir = await process_directory_with_preset_hash(dir1)

    # Create second file with DIFFERENT content but SAME hash
    content2 = "Different content!\nThis is file version 2 with totally different text"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(content2)
        temp_path2 = temp_file.name

    file2 = await File.from_local(temp_path2, hash_method=hash_value)  # Same hash!

    # Create second directory with DIFFERENT content but SAME hash
    with tempfile.TemporaryDirectory() as temp_dir2:
        (Path(temp_dir2) / "different1.txt").write_text("Totally different content")
        (Path(temp_dir2) / "different2.txt").write_text("Another different file")
        (Path(temp_dir2) / "subdir").mkdir()
        (Path(temp_dir2) / "subdir" / "nested.txt").write_text("Nested file")
        dir2 = await Dir.from_local(temp_dir2, dir_cache_key="dir_hash_67890")  # Same hash!

    # Process both again - these should hit cache (same random numbers)
    print("=== Second run (should hit cache) ===")
    result2_file = await process_file_with_preset_hash(file2)
    result2_dir = await process_directory_with_preset_hash(dir2)

    # Verify cache hits by checking outputs are identical
    assert result1_file == result2_file, f"File cache miss! {result1_file} != {result2_file}"
    assert result1_dir == result2_dir, f"Dir cache miss! {result1_dir} != {result2_dir}"

    return f"Cache test passed! File: {result1_file} | Dir: {result1_dir}"


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    result = flyte.with_runcontext("local").run(demo_cache_behavior)
    print(f"\nFinal result: {result}")
