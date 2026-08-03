"""Example to download directory content with Dir.download_sync()"""

import os
import tempfile
from pathlib import Path

import flyte
from flyte.io import Dir

env = flyte.TaskEnvironment(name="dir_download_sync_repro")


@env.task
async def create_remote_directory() -> Dir:
    """Create a small local directory (with a nested subdirectory) and upload it to object storage."""
    temp_dir = tempfile.mkdtemp(prefix="flyte_eng26_937_")

    with open(os.path.join(temp_dir, "root.txt"), "w") as f:  # noqa: ASYNC230
        f.write("root level file")

    nested = os.path.join(temp_dir, "nested")
    os.makedirs(nested)
    with open(os.path.join(nested, "child.txt"), "w") as f:  # noqa: ASYNC230
        f.write("file in nested subdirectory")

    uploaded_dir = await Dir.from_local(temp_dir)
    print(f"Uploaded {temp_dir} to remote: {uploaded_dir.path}")
    return uploaded_dir


@env.task
def download_directory_sync(d: Dir) -> list[str]:
    """
    The failing path: a *sync* task receives a Dir input and calls ``download_sync()``.
    """
    local_path = d.download_sync()
    print(f"Downloaded dir sync to: {local_path}")

    downloaded = sorted(str(p.relative_to(local_path)) for p in Path(local_path).rglob("*") if p.is_file())
    print(f"Downloaded files: {downloaded}")

    # Sanity-check that the recursive contents actually made it to disk.
    assert (Path(local_path) / "root.txt").exists(), "root.txt missing after download_sync"
    assert (Path(local_path) / "nested" / "child.txt").exists(), "nested/child.txt missing after download_sync"
    return downloaded


@env.task
async def main() -> list[str]:
    remote_dir = await create_remote_directory()
    # Pass the Dir as an input to a sync task, which downloads it via download_sync().
    return download_directory_sync(d=remote_dir)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
