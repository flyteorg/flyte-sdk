# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
# ]
# ///
"""
Run a cleanup callback when a task is aborted.

When you abort a run in the UI (or via ``Run.abort()`` / ``Action.abort()``), Flyte
deletes the task's Kubernetes pod. Kubernetes then sends ``SIGTERM`` to the task
process and, after ``terminationGracePeriodSeconds``, ``SIGKILL``. Flyte does *not*
install a ``SIGTERM`` handler for your task, so by default the process is terminated
without running any ``finally`` blocks or context managers.

This example shows how to catch that ``SIGTERM`` yourself and run an arbitrary
callback -- here, writing a checkpoint to the object store, a side effect -- before
the process exits. Two things matter:

1. Register the handler on the *main thread*. ``async def`` tasks run on the main
   thread's event loop, so ``loop.add_signal_handler(...)`` works from inside the
   task. (Plain ``def`` tasks run in a worker thread, where ``signal.signal`` is not
   allowed -- for those, register the handler at module import time instead.)

2. Give the callback enough time to finish. The default grace period is 30s, which
   may be too short to upload a large checkpoint, so we raise
   ``termination_grace_period_seconds`` on the pod template.
"""

from __future__ import annotations

import asyncio
import signal

from kubernetes.client import V1Container, V1PodSpec

import flyte

# Raise the pod termination grace period so the abort callback has time to flush its
# side effects (e.g. upload a checkpoint) before Kubernetes escalates to SIGKILL.
pod_template = flyte.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary")],
        termination_grace_period_seconds=600,  # 10 minutes; default is 30s
    ),
)

env = flyte.TaskEnvironment(
    name="abort_callback",
    pod_template=pod_template,
    image=flyte.Image.from_uv_script(__file__, name="flyte"),
)


@env.task
async def train(n_steps: int = 1000, step_seconds: float = 2.0) -> str:
    """A long-running loop that checkpoints its progress when the task is aborted."""
    checkpoint = flyte.ctx().checkpoint
    assert checkpoint is not None  # always available inside a task

    # Resume from the previous attempt's checkpoint, if any.
    prev = await checkpoint.load()
    start = int(prev.read_bytes().decode()) if prev is not None else 0
    print(f"Starting training at step {start}")

    aborted = asyncio.Event()

    def on_abort() -> None:
        # Runs on the event loop the moment Flyte aborts the task (SIGTERM).
        # Keep the callback itself lightweight: flip a flag and let the loop below
        # perform the actual side effects. (If you'd rather do the work directly in
        # the callback, schedule a coroutine with `asyncio.ensure_future(...)`.)
        print("Abort received -- will checkpoint current progress and exit.")
        aborted.set()

    # An `async def` task runs on the main thread's event loop, so we can register the
    # handler from inside the task. Aborting the run deletes the pod, and Kubernetes
    # sends SIGTERM to this process, which triggers `on_abort`.
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, on_abort)
    except (NotImplementedError, RuntimeError):
        # e.g. running locally off the main thread; abort handling only applies in-cluster.
        print("Could not register a SIGTERM handler here; continuing without abort handling.")

    step = start
    for step in range(start, n_steps):
        if aborted.is_set():
            # ---- extra code to run on abort; side effects live here ----
            await checkpoint.save(f"{step}".encode())
            print(f"Checkpoint saved at step {step}. Exiting due to abort.")
            return f"aborted at step {step}"
        # ... real training work would go here ...
        await asyncio.sleep(step_seconds)
        await checkpoint.save(f"{step + 1}".encode())
        print(f"Completed step {step + 1}/{n_steps}")

    return f"completed all {n_steps} steps"


if __name__ == "__main__":
    import time

    flyte.init_from_config()
    run = flyte.run(train, n_steps=1000, step_seconds=2.0)
    print(run.url)

    # Wait until the task is actually running, let it take a few steps, then abort it
    # to trigger the on-abort checkpoint callback. (You could also just click "Abort"
    # in the UI instead of the code below.)
    run.wait(wait_for="running")
    time.sleep(20)
    print("Aborting the run to trigger the on-abort checkpoint callback...")
    run.abort()
    print("Aborted. Inspect the task logs -- it should have saved a checkpoint before exiting.")
