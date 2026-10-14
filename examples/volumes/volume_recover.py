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
Volume checkpointing via ``@flyte.trace``.

``Volume`` lives in ``flyteplugins.union.io``, released to PyPI as
``flyteplugins-union`` (>=0.10.9). The platform wheels bundle the juicefs
binary, so the remote container image only needs a plain pip install.

The Volume type no longer ships built-in periodic checkpointing or crash
recovery. This example reconstructs both with ``@flyte.trace``: a single
long-running task mounts a writable volume once and runs one traced
``run_epoch`` span per epoch. Each span commits the volume — a durable,
immutable ``ROVolume`` snapshot — and returns it, so the run-details graph shows
a ``version -> version`` lineage chain, and a crashed attempt resumes from the
last committed epoch instead of restarting.

Three design points worth understanding:

**Mount once, commit many.** Mounting is expensive (FUSE format + mount), so we
do it exactly once per attempt. ``run_epoch`` does *not* mount on the happy path
— it calls :meth:`RWVolume.commit`, which drains writeback and uploads a fresh
metadata index while **keeping the mount live**, so a checkpoint costs one index
upload, not a remount.

**No Volume crosses the trace boundary — it's captured, not passed.**
``@flyte.trace`` serializes a function's *inputs* (to hash for the memo key) and
its *output* (for lineage). Passing a Volume either way is a trap: a live
``RWVolume`` would be drain+unmounted by the transformer's auto-finalize during
input hashing, and even an immutable ``ROVolume`` doesn't serialize identically
across the record/restore boundary (the transformer stamps
``produced_by.output_name`` onto the recorded output but not onto the live
value), so threading it would change the input hash on a retry and break
memoization. So ``run_epoch`` takes **only the ``epoch`` int** — a stable key —
and operates on the writable ``vol`` captured from the enclosing scope. The
``version -> version`` lineage still threads through each committed Volume's own
``parent_produced_by``.

**Crash recovery is the trace memoization.** On a retry, every ``run_epoch`` span
that already succeeded is replayed from its recorded output without re-running,
so the first epoch that actually executes is the one past the last checkpoint.
That epoch forks the last checkpoint into a writable working copy (``fork`` is
the blessed RO->RW path and shares its chunks copy-on-write), resuming exactly
where the crash hit. The example forces this path once via ``CRASH_AT_EPOCH``.

Prereqs:
- A bucket the cluster's service account can read/write.
- The Union mount broker on the cluster (its DaemonSet plus the
  ``volumes.union.ai`` CSIDriver) — that is what ``allow_volumes()`` binds
  to. No privileged pods and no ``/dev/fuse`` in the container.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from flyteplugins.union.io import ROVolume, Volume, allow_volumes

import flyte

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("volume-checkpoint-demo")

VOL_NAME = os.environ.get("VOL_NAME", "checkpoint-demo")
EPOCHS = int(os.environ.get("EPOCHS", "6"))
SECONDS_PER_EPOCH = float(os.environ.get("SECONDS_PER_EPOCH", "3"))
# Force a one-time crash after this epoch, on the first attempt only, to exercise
# the resume-from-checkpoint path. Set to a negative value to disable.
CRASH_AT_EPOCH = int(os.environ.get("CRASH_AT_EPOCH", "2"))

# A plain image + the released flyteplugins-union package is all Volumes need
# (juicefs is bundled in the PyPI platform wheels; the default sqlite store
# needs no extra daemon, and a brokered mount needs no fuse3 in the image).
image = (
    flyte.Image.from_debian_base(install_flyte=False, name="volume-checkpoint-demo")
    .with_pip_packages("flyteplugins-union>=0.10.9")
    .with_local_v2()  # last, so the local dev SDK wins over the pip layer's flyte dep
)

env = flyte.TaskEnvironment(
    name="volume-checkpoint-demo",
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


@env.task(retries=2)
async def train(volume_name: str, epochs: int) -> ROVolume:
    """Mount a writable volume once, then checkpoint one ``@flyte.trace`` span
    per epoch.

    Each epoch commits the volume (a durable ``ROVolume`` snapshot) and returns
    it, so on a retry the completed epochs are replayed from trace memoization
    and the loop resumes from the last checkpoint — see the module docstring.
    """
    logger.info("train: volume_name=%s epochs=%d", volume_name, epochs)
    # `vol` is the writable handle for this attempt — we mount it once and commit
    # *it* on every epoch (RWVolume.commit keeps the mount live). It starts as a
    # fresh empty volume; on a resume, run_epoch forks it from the checkpoint.
    vol = Volume.new(name=volume_name)
    # `current_vol` threads the latest immutable checkpoint through the loop and,
    # across a retry, via @flyte.trace memoization. `root` is process-local and
    # does double duty: still None means "not mounted yet", so the live mount is
    # stood up exactly once per attempt, and once set it holds the mount point
    # mount() resolved — a per-volume path keyed by name, which is what epochs
    # write under instead of a hardcoded directory.
    current_vol: Optional[ROVolume] = None
    root: Optional[Path] = None

    @flyte.trace
    async def run_epoch(epoch: int) -> ROVolume:
        nonlocal vol, root
        # The body only executes for NON-cached epochs, so on a retry this runs at
        # the first epoch past the last checkpoint. Stand the mount up once: if we
        # fast-forwarded to a checkpoint (held in the closure `current_vol`), fork
        # it into a writable working copy — fork() is the blessed RO->RW path, and
        # it shares the checkpoint's chunks copy-on-write so we resume its files.
        # The name is attempt-scoped so repeated retries don't collide. On a cold
        # start `current_vol` is None and `vol` is the empty Volume.new above.
        if root is None:
            if current_vol is not None:
                attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
                vol = await current_vol.fork(name=f"{volume_name}-a{attempt}")
            root = await vol.mount()
            logger.info("train: mounted at %s", root)
        logger.info("train: epoch %d/%d working...", epoch, epochs)
        await asyncio.sleep(SECONDS_PER_EPOCH)  # stand-in for real compute
        # Write this epoch's "checkpoint" artifact into the live mount.
        (root / f"ckpt-{epoch:03d}.bin").write_bytes(b"\x00" * 1024)
        with open(root / "epochs.log", "a") as f:  # noqa: ASYNC230
            f.write(f"epoch {epoch} done\n")
        # Commit the live RWVolume `vol`. RWVolume.commit drains writeback +
        # uploads a fresh index and keeps the mount live, returning the new
        # immutable version. (Never commit an ROVolume — it inherits the
        # deprecated Volume.commit(), which drains+UNMOUNTS.)
        return await vol.commit(message=f"epoch {epoch}")

    for epoch in range(epochs):
        current_vol = await run_epoch(epoch)
        # Force one hard failure after a committed epoch (first attempt only) so
        # the retry exercises the resume-from-checkpoint path. The completed
        # run_epoch spans above are memoized on retry; the loop fast-forwards
        # through them and the first live epoch remounts from the last commit.
        if epoch == CRASH_AT_EPOCH and int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0")) == 0:
            logger.error("train: simulating crash after epoch %d (attempt 0)", epoch)
            raise RuntimeError(f"simulated crash after epoch {epoch}")

    # finalize() drains writeback, unmounts, and publishes the terminal version.
    # Finalize the live writable handle `vol` — `current_vol` is the last
    # immutable ROVolume snapshot and has no finalize().
    sealed = await vol.finalize(message=f"trained {epochs} epochs")
    logger.info("train: sealed final index path=%s", sealed.index.path if sealed.index else None)
    return sealed


@env.task
async def verify(vol: ROVolume, expected_epochs: int) -> str:
    """Mount the sealed volume read-only and confirm every epoch landed."""
    logger.info("verify: mounting %s read-only", vol.name)
    root = await vol.mount()  # ROVolume always mounts read-only
    log = (root / "epochs.log").read_text()
    done = sum(1 for line in log.splitlines() if line.strip())
    logger.info("verify: %d epochs recorded (expected %d)", done, expected_epochs)
    if done != expected_epochs:
        raise AssertionError(f"epoch mismatch: got {done}, want {expected_epochs}")
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
