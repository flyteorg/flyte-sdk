import os
import tempfile
from pathlib import Path

import flyte
import flyte.io
import flyte.storage

env = flyte.TaskEnvironment(
    "file_dir_downloading_semantics",
)


@env.task
async def create_a_file() -> flyte.io.File:
    chunk_size = 1024  # 1 KiB
    with open("example.txt", "wb") as f:  # noqa: ASYNC230
        chunk = b"\0" * chunk_size
        for _ in range(512):  # 512 KiB
            f.write(chunk)

    uploaded_file_obj = await flyte.io.File.from_local("example.txt")
    return uploaded_file_obj


@env.task
async def download_a_file(f: flyte.io.File):
    # Download the file without specifying the local destination
    local_path = await f.download()
    print(f"With no local dest specified, downloaded original file at {f.path} to: {local_path}")

    # Download the file specifying the local destination
    local_path_with_dest = await f.download("new_parent/downloaded_example.txt")
    print(f"When specifying a non-existing path, downloaded file to: {local_path_with_dest}")

    # Download the file using an existing folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path_in_dir = await f.download(tmpdirname)
        print(f"When specifying an existing temp folder {tmpdirname=}, downloaded file to: {local_path_in_dir}")

    # Download the file using a non-existing path but one that ends in /
    local_path_with_trailing_slash = await f.download("new_parent2/")
    print(f"When specifying a non-existing path ending in /, downloaded file to: {local_path_with_trailing_slash}")


@env.task
async def create_a_dir() -> flyte.io.Dir:
    # Create a temporary directory with nested structure
    with tempfile.TemporaryDirectory(prefix="my_create_prefix", delete=False) as tmpdir:
        # Create some files in the root
        with open(os.path.join(tmpdir, "root_file.txt"), "w") as f:  # noqa: ASYNC230
            f.write("This is a root file\n" * 100)

        # Create nested directory with files
        nested_dir = os.path.join(tmpdir, "nested")
        os.makedirs(nested_dir)
        with open(os.path.join(nested_dir, "nested_file.txt"), "w") as f:  # noqa: ASYNC230
            f.write("This is a nested file\n" * 100)

        # Create another nested level
        nested2_dir = os.path.join(nested_dir, "nested2")
        os.makedirs(nested2_dir)
        with open(os.path.join(nested2_dir, "deep_file.txt"), "w") as f:  # noqa: ASYNC230
            f.write("This is a deeply nested file\n" * 100)

        # Upload the directory (If mode="local", this upload is skipped)
        uploaded_dir_obj = await flyte.io.Dir.from_local(tmpdir)
        return uploaded_dir_obj


@env.task
async def download_a_dir(d: flyte.io.Dir):
    # Download the directory without specifying the local destination
    # This will create a unique temp path and append the source directory name
    local_path = await d.download()
    # If mode="local", this copy is skipped
    print(f"With no local dest specified, downloaded original dir at {d.path} to: {local_path}")
    print(f"  Contents: {os.listdir(local_path)}")

    # Download the directory specifying a custom destination
    # The directory contents will go directly into this path
    local_path_with_dest = await d.download("new_parent/downloaded_dir")
    print(f"When specifying a non-existing path, downloaded dir to: {Path(local_path_with_dest).absolute()}")
    print(f"  Contents: {os.listdir(local_path_with_dest)}")

    # Download the directory using an existing folder
    # The directory contents will go directly into the existing folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path_in_dir = await d.download(tmpdirname)
        print(
            f"When specifying an existing temp folder {tmpdirname=},"
            f" downloaded dir to: {Path(local_path_in_dir).absolute()}"
        )
        print(f"  Contents: {os.listdir(local_path_in_dir)}")


@env.task
async def main():
    print("=" * 60)
    print("FILE EXAMPLES")
    print("=" * 60)
    a_file = await create_a_file()
    await download_a_file(a_file)

    print("\n" + "=" * 60)
    print("DIR EXAMPLES")
    print("=" * 60)
    a_dir = await create_a_dir()
    await download_a_dir(a_dir)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    # r = flyte.with_runcontext(mode="local").run(main)
    print(r.url)
