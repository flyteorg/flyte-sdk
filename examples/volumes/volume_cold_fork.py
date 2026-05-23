# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
#    "flyteplugins-union",
# ]
#
# [tool.uv.sources]
# flyteplugins-union = { path = "../../../../unionai/flyteplugins-union", editable = true }
# ///
"""
Cold-fork correctness example.

Exercises ``Volume.fork()`` on an *unmounted* Volume. The fork task never
calls ``mount()`` on the parent — it just calls ``parent.fork(...)``,
which downloads the index, snapshots it, and uploads. The fork task then
mounts the *fork* and reads the marker the parent wrote, proving that
cold-forked Volumes preserve the parent's namespace.

Workflow:

1. ``populate_parent`` mounts a fresh Volume, writes a marker, commits.
2. ``cold_fork`` receives the committed Volume, forks it *without
   mounting*, then mounts the fork and reads the marker back.
3. ``main`` chains them and returns the marker contents.
"""

import logging
import os
import uuid
from pathlib import Path

from flyteplugins.union.io.volume import Volume, with_volume_deps

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-cold-fork-demo")

VOL_NAME = os.environ.get("VOL_NAME", "cold-fork-demo")
MARKER_CONTENTS = "parent state — should be visible to cold fork\n"

IDL2 = "git+https://github.com/flyteorg/flyte.git@v2#subdirectory=gen/python"
base = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-cold-fork-demo")
    .with_apt_packages("git")
    .with_pip_packages(IDL2, "kubernetes")
)
image = with_volume_deps(base, install_flyte=False, install_local=True).with_local_v2()

env = flyte.TaskEnvironment(
    name="volume-cold-fork-demo",
    enable_fuse_mount=True,
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)


@env.task
async def populate_parent(volume_name: str) -> Volume:
    """Format + populate a parent Volume, then commit."""
    logger.info("populate_parent: name=%s", volume_name)
    parent = Volume.empty(name=volume_name)
    await parent.mount()
    Path("/workspace/marker.txt").write_text(MARKER_CONTENTS)
    return await parent.commit()


@env.task
async def cold_fork_and_read(parent: Volume, fork_name: str) -> str:
    """Fork *without* mounting the parent — the cold path — then mount the
    fork and read the marker.
    """
    logger.info("cold_fork_and_read: parent=%s -> fork=%s (no parent mount)", parent.name, fork_name)
    # Critical: no parent.mount() call. fork() must handle the cold path.
    forked = await parent.fork(name=fork_name)
    logger.info("cold_fork_and_read: forked, new index path=%s", forked.index.path if forked.index else None)

    await forked.mount()
    contents = Path("/workspace/marker.txt").read_text()
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
