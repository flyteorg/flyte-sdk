# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
JuiceFS Volume example.

NOTE: ``Volume`` and ``juicefs_image`` live in ``flyte.extras`` and have not yet
been released to PyPI. The remote container image is built from the locally
checked-out flyte wheel via ``Image.with_local_v2()``. Build the wheel first::

    make dist

Demonstrates a JuiceFS-backed ``Volume`` flowing through a multi-task workflow:

1. ``init_volume`` formats a fresh JuiceFS namespace, writes a file, commits
   the SQLite index back to blob storage, and returns a ``Volume``.
2. ``append`` mounts the committed volume, appends to the file, commits, and
   returns the new ``Volume``.
3. ``branch`` mounts a volume, calls ``Volume.fork()`` to snapshot the index,
   and returns the fork — a cheap, namespace-isolated copy that shares chunks
   in S3.
4. ``main`` chains them together so the run produces a lineage:
   ``init -> appended -> fork``.

Prereqs:
- A bucket the cluster's service account can read/write.
- Cluster nodes expose ``/dev/fuse``; pods can run privileged with CAP_SYS_ADMIN
  so the primary container can FUSE-mount JuiceFS in-process.
"""

import logging
import os
import subprocess
from pathlib import Path

import flyte
from flyte.extras import Volume, juicefs_image, juicefs_pod_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("juicefs-demo")


@flyte.trace
def _snapshot(tag: str) -> None:
    """Dump a listing of the workspace and meta dir, plus the on-disk size of
    the sqlite index. Helps pinpoint where state diverges across the lineage.
    """
    try:
        ws = subprocess.run(["ls", "-la", "/workspace"], capture_output=True, text=True).stdout
    except Exception as e:  # noqa: BLE001
        ws = f"<error: {e}>"
    try:
        meta = subprocess.run(["ls", "-la", "/var/jfs"], capture_output=True, text=True).stdout
    except Exception as e:  # noqa: BLE001
        meta = f"<error: {e}>"
    db_size = -1
    try:
        db_size = os.path.getsize("/var/jfs/juicefs.db")
    except OSError:
        pass
    # Is /workspace actually a FUSE mountpoint from the primary's view?
    mtype = "<unknown>"
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5 and parts[4] == "/workspace":
                    mtype = parts[8] if len(parts) > 8 else "<short>"
                    mtype = f"fs={mtype} line={line.strip()}"
                    break
    except OSError as e:
        mtype = f"<mountinfo err: {e}>"
    logger.info(
        "[%s] db_bytes=%d /workspace mount: %s\n--- /workspace ---\n%s--- /var/jfs ---\n%s",
        tag, db_size, mtype, ws, meta,
    )

JFS_BUCKET = os.environ.get(
    "JFS_BUCKET",
    "https://s3.us-east-2.amazonaws.com/union-oc-production-demo/juicefs",
)
JFS_NAME = os.environ.get("JFS_NAME", "testfs")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

POD = juicefs_pod_template()

# Bake the locally-built flyte wheel + juicefs binary into the image. Run
# `make dist` at the repo root first.
IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
base = (
    flyte.Image.from_debian_base(install_flyte=False, name="juicefs-demo")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = juicefs_image(base).with_local_v2()

env = flyte.TaskEnvironment(
    name="juicefs-demo",
    pod_template=POD,
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)


@env.task(cache="auto")
async def init_volume(volume_name: str) -> Volume:
    logger.info("init_volume: declaring fresh volume name=%s bucket=%s", volume_name, JFS_BUCKET)
    vol = Volume.empty(name=volume_name, bucket=JFS_BUCKET)
    await vol.mount()
    _snapshot("init_volume:post-mount")
    Path("/workspace/hello.txt").write_text("hello from juicefs\n")
    _snapshot("init_volume:post-write")
    committed = await vol.commit()
    _snapshot("init_volume:post-commit")
    logger.info("init_volume: committed, index path=%s", committed.index.path if committed.index else None)
    return committed


@env.task
async def append(vol: Volume, line: str) -> Volume:
    logger.info("append: mounting volume name=%s (index=%s)", vol.name, vol.index.path if vol.index else None)
    await vol.mount()
    _snapshot("append:post-mount")
    with open("/workspace/hello.txt", "a") as f:  # noqa: ASYNC230
        f.write(line + "\n")
    _snapshot("append:post-write")
    committed = await vol.commit()
    _snapshot("append:post-commit")
    logger.info("append: committed, new index path=%s", committed.index.path if committed.index else None)
    return committed


@env.task
async def branch(vol: Volume, name: str) -> Volume:
    logger.info("branch: mounting source volume name=%s to fork into name=%s", vol.name, name)
    await vol.mount()
    _snapshot("branch:post-mount")
    forked = await vol.fork(name=name)
    _snapshot("branch:post-fork")
    logger.info("branch: forked, new index path=%s", forked.index.path if forked.index else None)
    return forked


@env.task
async def read_all(vol: Volume) -> str:
    logger.info("read_all: mounting volume name=%s (index=%s)", vol.name, vol.index.path if vol.index else None)
    await vol.mount()
    _snapshot("read_all:post-mount")
    contents = Path("/workspace/hello.txt").read_text()
    logger.info("read_all: read %d bytes", len(contents))
    return contents


@env.task
async def main() -> str:
    import uuid

    # Unique volume name per run so `juicefs format` doesn't collide with
    # chunks left in the bucket by earlier runs.
    run_name = f"{JFS_NAME}-{uuid.uuid4().hex[:8]}"
    logger.info("main: starting juicefs lineage demo with name=%s", run_name)
    v0 = await init_volume(volume_name=run_name)
    logger.info("main: v0 ready, appending on main branch")
    v1 = await append(v0, line="appended on main branch")
    logger.info("main: v1 ready, forking")
    fork = await branch(v1, name=f"{run_name}-fork")
    logger.info("main: fork ready, appending only on the fork")
    fork_after = await append(fork, line="only on the fork")
    logger.info("main: reading back from fork")
    result = await read_all(fork_after)
    logger.info("main: done, fork content len=%d", len(result))
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
