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
Volume checkpoint + crash-recovery example.

NOTE: ``Volume`` lives in ``flyteplugins.union.io`` and has not yet been
released to PyPI. The remote container image bakes the locally-built
``flyteplugins-union`` wheel via ``with_local_flyteplugins_union``. Build a
pod-compatible wheel from a checkout of the flyteplugins-union repo first::

    make dist-bundled PLATFORM=linux-amd64

Demonstrates the long-running-task pattern: a single task holds a mounted
``RWVolume`` for the duration of a multi-epoch "training" loop and lets the
Volume checkpoint itself in the background.

Two related-but-distinct mechanisms are at play, both driven by the single
``checkpoint_interval_seconds`` argument to ``mount()``:

1. **Crash recovery (automatic).** Each background checkpoint drains the
   writeback queue, uploads a fresh metadata index *without unmounting*, and
   writes the resulting Volume to the task's Flyte checkpoint path. If the pod
   is killed mid-task and Flyte retries the action, ``mount()`` finds that
   snapshot via ``prev_checkpoint_path`` and resumes from it — superseding the
   originally-declared index. A hard kill therefore loses at most
   ``checkpoint_interval_seconds`` of writes, even across a full pod reload.
   This needs no callback and no code beyond setting the interval.

2. **Lineage visibility (opt-in).** Passing ``on_checkpoint`` runs a callback
   with each checkpointed Volume. ``report_checkpoint_trace`` is a ready-made
   one: it records a ``@flyte.trace`` span per checkpoint so every intermediate
   version shows up in the run-details graph, even if the task never reaches
   its final ``return``.

Contrast with the sibling ``volume_example.py``, where each *task* forks /
finalizes a volume so versions flow through task signatures. Here a *single*
task checkpoints in place, which is the right shape for an epoch loop (UC2).

Prereqs:
- A bucket the cluster's service account can read/write.
- Cluster nodes expose ``/dev/fuse``; pods can run privileged with
  CAP_SYS_ADMIN so the primary container can FUSE-mount in-process.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

from flyteplugins.union.io import ROVolume, Volume, report_checkpoint_trace
from flyteplugins.union.utils.image import with_local_flyteplugins_union

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-checkpoint-demo")

VOL_NAME = os.environ.get("VOL_NAME", "checkpoint-demo")
EPOCHS = int(os.environ.get("EPOCHS", "6"))
# How long each "epoch" takes. The background loop checkpoints every
# CHECKPOINT_INTERVAL seconds, so keep epochs longer than the interval to see
# multiple checkpoints land per run.
SECONDS_PER_EPOCH = float(os.environ.get("SECONDS_PER_EPOCH", "20"))
CHECKPOINT_INTERVAL = float(os.environ.get("CHECKPOINT_INTERVAL", "10"))
# Crash on this epoch (0-indexed) the *first* time we reach it, to exercise the
# automatic recovery path. Set to a negative value to disable.
CRASH_ON_EPOCH = int(os.environ.get("CRASH_ON_EPOCH", "3"))

# A plain image + the flyteplugins-union wheel is all Volumes need (juicefs is
# bundled in the wheel; sqlite mounts via raw syscalls under enable_fuse_mount).
# `with_local_flyteplugins_union` bakes the locally-built wheel for dev
# iteration — run `make dist-bundled PLATFORM=linux-amd64` in the
# flyteplugins-union repo first.
base = flyte.Image.from_debian_base(install_flyte=False, name="volume-checkpoint-demo").with_local_v2()
image = with_local_flyteplugins_union(base)

env = flyte.TaskEnvironment(
    name="volume-checkpoint-demo",
    # CAP_SYS_ADMIN + /dev/fuse come from this flag; no PodTemplate needed.
    enable_fuse_mount=True,
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)


def _completed_epochs(state_path: Path) -> int:
    """Read how many epochs the volume already has on disk.

    After a crash-and-recover, the mount comes back from the last checkpoint,
    so this reflects the recovered state — the loop picks up where it left off
    instead of restarting from epoch 0.
    """
    if not state_path.exists():
        return 0
    return sum(1 for line in state_path.read_text().splitlines() if line.strip())


@env.task(retries=3)
async def train(volume_name: str, epochs: int) -> ROVolume:
    """Long-running task that checkpoints a mounted volume every interval.

    The volume is mounted once and held for the whole loop. The background
    checkpoint loop (enabled by ``checkpoint_interval_seconds``) keeps the
    on-object-store index fresh, so a pod kill + Flyte retry resumes from the
    last checkpoint rather than losing the run.
    """
    attempt = flyte.ctx().attempt_number
    logger.info("train: volume_name=%s epochs=%d attempt=%d", volume_name, epochs, attempt)

    # Reuse the same volume name across attempts so a retry's mount() can find
    # the checkpoint the crashed attempt wrote under that name.
    vol = Volume.new(name=volume_name)
    await vol.mount(
        mount_path="/workspace",
        # Enables the background checkpoint loop: drain + snapshot every N
        # seconds, write each snapshot to the Flyte checkpoint path (this is
        # what makes the volume crash-recoverable), and...
        checkpoint_interval_seconds=CHECKPOINT_INTERVAL,
        # ...also surface each checkpoint in the lineage graph as a trace span.
        # Purely for visibility; recovery above works without it.
        on_checkpoint=report_checkpoint_trace,
    )

    state = Path("/workspace/epochs.log")
    start = _completed_epochs(state)
    if start > 0:
        logger.warning("train: recovered %d completed epochs from checkpoint — resuming", start)

    for epoch in range(start, epochs):
        # Simulate a hard pod failure exactly once. On the retry, mount() above
        # restores the checkpoint, _completed_epochs() reports the recovered
        # count, and the loop resumes — no work before the last checkpoint redone.
        if epoch == CRASH_ON_EPOCH and attempt == 0:
            logger.error("train: simulating pod crash at epoch %d (attempt 0)", epoch)
            raise RuntimeError(f"simulated crash at epoch {epoch}")

        logger.info("train: epoch %d/%d working...", epoch, epochs)
        await asyncio.sleep(SECONDS_PER_EPOCH)  # stand-in for real compute
        # Write the "checkpoint" for this epoch. The background loop uploads it
        # to object storage and to the Flyte checkpoint path on its own cadence.
        with open(state, "a") as f:  # noqa: ASYNC230
            f.write(f"epoch {epoch} done\n")
        Path(f"/workspace/ckpt-{epoch:03d}.bin").write_bytes(b"\x00" * 1024)

    # finalize() drains writeback, stops the checkpoint loop, unmounts, and
    # publishes the terminal immutable ROVolume.
    sealed = await vol.finalize(message=f"trained {epochs} epochs")
    logger.info("train: sealed final index path=%s", sealed.index.path if sealed.index else None)
    return sealed


@env.task
async def verify(vol: ROVolume, expected_epochs: int) -> str:
    """Mount the sealed volume read-only and confirm every epoch survived."""
    logger.info("verify: mounting %s read-only", vol.name)
    await vol.mount()  # ROVolume always mounts read-only
    log = Path("/workspace/epochs.log").read_text()
    done = sum(1 for line in log.splitlines() if line.strip())
    logger.info("verify: %d epochs recorded (expected %d)", done, expected_epochs)
    if done != expected_epochs:
        raise AssertionError(f"epoch mismatch after recovery: got {done}, want {expected_epochs}")
    return log


@env.task
async def main() -> str:
    run_name = f"{VOL_NAME}-{uuid.uuid4().hex[:8]}"
    logger.info("main: starting checkpoint demo name=%s epochs=%d", run_name, EPOCHS)
    trained = await train(volume_name=run_name, epochs=EPOCHS)
    return await verify(trained, expected_epochs=EPOCHS)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
