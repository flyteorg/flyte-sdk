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

Demonstrates the typed Volume lifecycle (PRD §Core Concepts) flowing
through a multi-task workflow. Tasks exchange immutable ``ROVolume``
values; to mutate, a task forks one into a writable ``RWVolume``, writes,
and ``finalize()``s back to an ``ROVolume``:

1. ``init_volume`` creates a fresh ``RWVolume`` (``Volume.new``), writes a
   file, and ``finalize()``s — drains writeback, unmounts, and publishes
   an immutable ``ROVolume``.
2. ``append`` ``fork()``s the incoming ``ROVolume`` into an ``RWVolume``,
   appends to the file, and ``finalize()``s the new immutable version.
3. ``branch`` ``fork()``s a volume into a named writable copy and
   ``finalize()``s it — a cheap, namespace-isolated snapshot that shares
   chunks in object storage.
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

from flyteplugins.union.io.volume import ROVolume, Volume, with_volume_deps

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
async def init_volume(volume_name: str) -> ROVolume:
    logger.info("init_volume: declaring fresh volume name=%s", volume_name)
    # Pin to sqlite: this lineage forks (see `branch`), and the package
    # default (badger) doesn't support fork(). sqlite is daemon-less, so it
    # works with the lean `with_volume_deps` image below. Volume.new()
    # returns a writable RWVolume.
    vol = Volume.new(name=volume_name, metadata_store_type="sqlite")
    logger.info("init_volume: bucket resolved to %s", vol.bucket)
    await vol.mount()
    Path("/workspace/hello.txt").write_text("hello from volume\n")
    # finalize() drains writeback, unmounts, and publishes an immutable
    # ROVolume. (Returning the mounted RWVolume directly would do the same
    # via the type transformer.)
    sealed = await vol.finalize(message="initial write")
    logger.info("init_volume: sealed, index path=%s", sealed.index.path if sealed.index else None)
    return sealed


@env.task
async def append(vol: ROVolume, line: str) -> ROVolume:
    logger.info("append: forking immutable volume name=%s into a writable copy", vol.name)
    # An ROVolume is immutable; fork it into an RWVolume to write.
    rw = await vol.fork(name=f"{vol.name}-app")
    await rw.mount()
    with open("/workspace/hello.txt", "a") as f:  # noqa: ASYNC230
        f.write(line + "\n")
    sealed = await rw.finalize(message=f"append: {line!r}")
    logger.info("append: sealed, new index path=%s", sealed.index.path if sealed.index else None)
    return sealed


@env.task
async def branch(vol: ROVolume, name: str) -> ROVolume:
    logger.info("branch: forking source volume name=%s into name=%s", vol.name, name)
    rw = await vol.fork(name=name)
    await rw.mount()
    sealed = await rw.finalize(message="branch point")
    logger.info("branch: sealed fork, new index path=%s", sealed.index.path if sealed.index else None)
    return sealed


@env.task
async def read_all(vol: ROVolume) -> str:
    logger.info("read_all: mounting volume name=%s (read-only)", vol.name)
    await vol.mount()  # ROVolume always mounts read-only
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
