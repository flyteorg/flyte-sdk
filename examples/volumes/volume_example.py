# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
#    "flyteplugins-union",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# flyteplugins-union = { path = "../../../../unionai/flyteplugins-union", editable = true }
# ///
"""
Volume example.

NOTE: ``Volume`` and ``with_volume_deps`` live in ``flyteplugins.union.io.volume``
and have not yet been released to PyPI. The remote container image bakes the
locally-built ``flyteplugins-union`` wheel via ``with_volume_deps(install_local=True)``.
Build the wheel from a checkout of the flyteplugins-union repo first::

    make dist

Demonstrates a ``Volume`` flowing through a multi-task workflow:

1. ``init_volume`` formats a fresh namespace, writes a file, commits the
   index back to blob storage, and returns a ``Volume``.
2. ``append`` mounts the committed volume, appends to the file, commits,
   and returns the new ``Volume``.
3. ``branch`` mounts a volume, calls ``Volume.fork()`` to snapshot the
   index, and returns the fork — a cheap, namespace-isolated copy that
   shares chunks in object storage.
4. ``main`` chains them together so the run produces a lineage:
   ``init -> appended -> fork``.

Prereqs:
- A bucket the cluster's service account can read/write.
- Cluster nodes expose ``/dev/fuse``; pods can run privileged with
  CAP_SYS_ADMIN so the primary container can FUSE-mount in-process.
"""

import logging
import os
import uuid
from pathlib import Path

from flyteplugins.union.io.volume import Volume, with_volume_deps

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-demo")


VOL_NAME = os.environ.get("VOL_NAME", "demo-vol")

# Bake the locally-built flyteplugins-union wheel + volume runtime deps
# into a custom base. Run `make dist-bundled PLATFORM=linux-amd64` in the flyteplugins-union repo first.
base = flyte.Image.from_debian_base(install_flyte=False, name="volume-demo").with_local_v2()
image = with_volume_deps(base, install_local=True)

env = flyte.TaskEnvironment(
    name="volume-demo",
    # CAP_SYS_ADMIN + /dev/fuse come from this flag; no PodTemplate needed.
    enable_fuse_mount=True,
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)


@env.task(cache="auto")
async def init_volume(volume_name: str) -> Volume:
    logger.info("init_volume: declaring fresh volume name=%s", volume_name)
    vol = Volume.empty(name=volume_name)
    logger.info("init_volume: bucket resolved to %s", vol.bucket)
    await vol.mount()
    Path("/workspace/hello.txt").write_text("hello from volume\n")
    committed = await vol.commit()
    logger.info("init_volume: committed, index path=%s", committed.index.path if committed.index else None)
    return committed


@env.task
async def append(vol: Volume, line: str) -> Volume:
    logger.info("append: mounting volume name=%s", vol.name)
    await vol.mount()
    with open("/workspace/hello.txt", "a") as f:  # noqa: ASYNC230
        f.write(line + "\n")
    committed = await vol.commit()
    logger.info("append: committed, new index path=%s", committed.index.path if committed.index else None)
    return committed


@env.task
async def branch(vol: Volume, name: str) -> Volume:
    logger.info("branch: mounting source volume name=%s to fork into name=%s", vol.name, name)
    await vol.mount()
    forked = await vol.fork(name=name)
    logger.info("branch: forked, new index path=%s", forked.index.path if forked.index else None)
    return forked


@env.task
async def read_all(vol: Volume) -> str:
    logger.info("read_all: mounting volume name=%s", vol.name)
    await vol.mount()
    contents = Path("/workspace/hello.txt").read_text()
    logger.info("read_all: read %d bytes", len(contents))
    return contents


@env.task
async def main() -> str:
    # Unique volume name per run so the format step doesn't collide with
    # chunks left in the bucket by earlier runs.
    run_name = f"{VOL_NAME}-{uuid.uuid4().hex[:8]}"
    logger.info("main: starting volume lineage demo with name=%s", run_name)
    v0 = await init_volume(volume_name=run_name)
    v1 = await append(v0, line="appended on main branch")
    fork = await branch(v1, name=f"{run_name}-fork")
    fork_after = await append(fork, line="only on the fork")
    result = await read_all(fork_after)
    logger.info("main: done, fork content len=%d", len(result))
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
