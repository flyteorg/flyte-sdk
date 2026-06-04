# /// script
# requires-python = "==3.12"
# dependencies = [
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

NOTE: ``Volume`` lives in ``flyteplugins.union.io`` and has not yet been
released to PyPI. The remote container image bakes the locally-built
``flyteplugins-union`` wheel via ``with_local_flyteplugins_union``. Build a
pod-compatible wheel from a checkout of the flyteplugins-union repo first::

    make dist-bundled PLATFORM=linux-amd64

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

from flyteplugins.union.io import ROVolume, Volume

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-demo")


VOL_NAME = os.environ.get("VOL_NAME", "demo-vol")

# A plain image + the flyteplugins-union wheel is all Volumes need (juicefs is
# bundled in the wheel; the default sqlite store mounts via raw syscalls under
# enable_fuse_mount). `with_local_flyteplugins_union` bakes the locally-built
# wheel for dev iteration — run `make dist-bundled PLATFORM=linux-amd64` in the
# flyteplugins-union repo first.
base = flyte.Image.from_debian_base(install_flyte=False, name="volume-demo").with_local_v2()
# image = with_local_flyteplugins_union(base)
image = base.with_pip_packages("flyteplugins-union>=0.4.0")

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
    import os

    logger.info("AZURE creds in pod:")
    for k in os.environ:
        if "AZURE" in k.upper():
            logger.info("%s", k)

    # Volume.new() returns a writable RWVolume backed by the default sqlite
    # store — daemon-less and fork-capable, so the later `branch`/`append`
    # forks Just Work with no extra image deps.
    vol = Volume.new(name=volume_name)
    logger.info("init_volume: bucket resolved to %s", vol.bucket)
    # mount_path is configurable; "/workspace" is the default.
    await vol.mount(mount_path="/workspace")
    Path("/workspace/hello.txt").write_text("hello from volume\n")
    # finalize() drains writeback, unmounts, and returns an immutable ROVolume.
    # Use commit() instead to flush without unmounting (keeps the volume live).
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
