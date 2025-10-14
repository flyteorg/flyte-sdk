import tempfile

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
async def main():
    a_file = await create_a_file()
    await download_a_file(a_file)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
