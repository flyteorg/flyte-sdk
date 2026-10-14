# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-union>=0.10.9",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Cold-fork correctness example.

Exercises ``Volume.fork()`` on an *unmounted* Volume. The fork task never
calls ``mount()`` on the parent — it just calls ``parent.fork(...)``,
which downloads the index, snapshots it, and uploads. The fork task then
mounts the *fork* and reads the marker the parent wrote, proving that
cold-forked Volumes preserve the parent's namespace.

Workflow:

1. ``populate_parent`` creates a fresh ``RWVolume``, writes a marker, and
   ``finalize()``s it into an immutable ``ROVolume``.
2. ``cold_fork`` receives the sealed ``ROVolume``, forks it *without
   mounting* (``ROVolume.fork() -> RWVolume``), then mounts the fork and
   reads the marker back.
3. ``main`` chains them and returns the marker contents.
"""

import logging
import os
import uuid

from flyteplugins.union.io import ROVolume, Volume, allow_volumes

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-cold-fork-demo")

VOL_NAME = os.environ.get("VOL_NAME", "cold-fork-demo")
MARKER_CONTENTS = "parent state — should be visible to cold fork\n"

# A plain image + the released flyteplugins-union package is all Volumes need
# (juicefs is bundled in the PyPI platform wheels; a brokered mount needs no
# fuse3 in the image).
image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-cold-fork-demo")
    .with_pip_packages("flyteplugins-union>=0.10.9")
    .with_local_v2()  # last, so the local dev SDK wins over the pip layer's flyte dep
)

env = flyte.TaskEnvironment(
    name="volume-cold-fork-demo",
    # Zero-privilege Volume mounts via the node mount broker: the pod adopts a
    # premounted FUSE channel's file descriptor over a socket instead of
    # calling mount(2), so it needs no CAP_SYS_ADMIN, no /dev/fuse, and no
    # hostPath. Requires the broker DaemonSet + volumes.union.ai CSIDriver.
    pod_template=allow_volumes(),
    image=image,
    # The mount client draws real memory: JuiceFS plus its metadata store, and
    # the memory-backed passthrough staging emptyDir allow_volumes() attaches
    # (tmpfs pages count against the pod's limit). 1Gi is under the floor — a
    # mount-then-cold-fork pod gets OOMKilled there.
    resources=flyte.Resources(cpu="1", memory="4Gi"),
)


@env.task
async def populate_parent(volume_name: str) -> ROVolume:
    """Format + populate a parent volume, then seal it into an ROVolume."""
    logger.info("populate_parent: name=%s", volume_name)
    # Default sqlite store is daemon-less and fork-capable, so the later
    # cold fork (see `cold_fork_and_read`) works with no extra image deps.
    # Volume.new() returns a writable RWVolume.
    parent = Volume.new(name=volume_name)
    # mount() returns the resolved mount point; unset, mount_path defaults to a
    # per-volume path keyed by name. Read and write under what it hands back
    # rather than hardcoding a directory.
    root = await parent.mount()
    logger.info("populate_parent: mounted at %s", root)
    (root / "marker.txt").write_text(MARKER_CONTENTS)
    return await parent.finalize(message="parent marker")


@env.task
async def cold_fork_and_read(parent: ROVolume, fork_name: str) -> str:
    """Fork *without* mounting the parent — the cold path — then mount the
    fork and read the marker.
    """
    logger.info("cold_fork_and_read: parent=%s -> fork=%s (no parent mount)", parent.name, fork_name)
    # Critical: no parent.mount() call. fork() must handle the cold path.
    forked = await parent.fork(name=fork_name)
    logger.info("cold_fork_and_read: forked, new index path=%s", forked.index.path if forked.index else None)

    root = await forked.mount()
    contents = (root / "marker.txt").read_text()
    logger.info("cold_fork_and_read: read %d bytes from fork", len(contents))
    if contents != MARKER_CONTENTS:
        raise AssertionError(f"cold fork lost parent state: got {contents!r}, want {MARKER_CONTENTS!r}")
    return contents


@env.task
async def main() -> str:
    run_id = uuid.uuid4().hex[:8]
    parent = await populate_parent(volume_name=f"{VOL_NAME}-{run_id}")
    return await cold_fork_and_read(parent, fork_name=f"{VOL_NAME}-{run_id}-fork")


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
