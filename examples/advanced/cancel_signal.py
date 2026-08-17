"""Cancel a running pipeline gracefully via a signal (condition).

On startup the pipeline task creates a *condition* — a signal that an external
actor can fire at any time — and then races it against its real work:

1. ``flyte.new_condition("cancel", ...)`` registers the signal with the backend
   the moment the task starts, so it is fireable for the task's whole lifetime.
2. The task wraps ``condition.wait()`` and the actual work (a couple of slow
   downstream tasks) in ``asyncio.Task``s and races them with
   ``asyncio.wait(..., return_when=FIRST_COMPLETED)``.
3. If the signal fires first, the task cancels the work. Cancelling the asyncio
   task that wraps a child-task call makes the controller abort that child
   action server-side too, so the in-flight downstream tasks are torn down.
   The pipeline then exits *gracefully*: it logs that it was canceled and
   returns a normal "canceled" result — the run itself SUCCEEDS.
4. If the work finishes first, the watcher is retired and the pipeline returns
   its results as usual.

Contrast with ``abort_callback.py``: aborting a run kills the pod (SIGTERM →
SIGKILL) and the run ends ABORTED. Here nothing is killed — cancellation is
cooperative and application-level, and the run completes successfully with a
result that says it was canceled.

Run it remotely::

    python cancel_signal.py            # happy path: no signal, pipeline completes
    python cancel_signal.py cancel     # fires the signal shortly after launch

Or fire the signal yourself while the first form is running::

    flyte signal condition <run-name> <action-name> true
"""

from __future__ import annotations

import asyncio

import flyte

env = flyte.TaskEnvironment(name="cancel_signal")


@env.task
async def process(part: int, work_seconds: float = 30.0) -> str:
    """A slow downstream task — sleeps in small steps so there is a window to cancel it."""
    steps = max(1, int(work_seconds))
    for step in range(steps):
        await asyncio.sleep(work_seconds / steps)
        flyte.logger.info(f"part {part}: step {step + 1}/{steps} done")
    return f"part {part} done"


@env.task
async def pipeline(work_seconds: float = 30.0) -> str:
    """Race a cancel signal against a couple of slow downstream tasks."""
    # 1. On startup, create the cancel signal. From this point on an external
    #    actor (UI, CLI, or remote.Condition.signal) can fire it at any time.
    cancel_signal = await flyte.new_condition.aio(
        "cancel",
        prompt="Fire with `true` to cancel the pipeline gracefully.",
        data_type=bool,
    )
    watcher = asyncio.create_task(cancel_signal.wait.aio())

    # 2. Kick off the real work: two downstream tasks that take a while.
    #    gather() schedules them immediately and returns a cancellable future.
    work = asyncio.gather(
        process(part=1, work_seconds=work_seconds),
        process(part=2, work_seconds=work_seconds),
    )

    # 3. Race them: whichever finishes first decides the pipeline's fate.
    await asyncio.wait({watcher, work}, return_when=asyncio.FIRST_COMPLETED)

    # Note: firing the signal with `false` also completes the wait; we treat
    # only a `true` payload as a cancellation request.
    if watcher.done() and watcher.result():
        # Signal fired -> cancel the in-flight work. Cancelling the asyncio task
        # wrapping the child-task calls makes the controller abort those child
        # actions server-side as well, so nothing keeps running behind our back.
        work.cancel()
        try:
            await work
        except asyncio.CancelledError:
            pass
        # Graceful exit: the run still SUCCEEDS, with a result saying why.
        flyte.logger.info("Cancel signal received -- pipeline canceled gracefully.")
        return "canceled: cancel signal was fired"

    # Work finished first (or the signal came in as `false`) -> retire the
    # watcher and return the results.
    watcher.cancel()
    results = await work
    return f"completed: {results}"


if __name__ == "__main__":
    import sys
    import time

    import flyte.remote as remote

    flyte.init_from_config(log_level="INFO")

    fire_signal = "cancel" in sys.argv[1:]

    r = flyte.run(pipeline)
    print("run url:", r.url)

    # Wait for the `cancel` condition action to show up, then tell the user how
    # to fire it (or fire it ourselves when invoked as `python cancel_signal.py cancel`).
    cond = None
    while cond is None:
        for c in remote.Condition.listall(run_name=r.name):
            # A condition waiting for a signal sits in PAUSED (briefly RUNNING first).
            if c.phase.removeprefix("ACTION_PHASE_") in ("RUNNING", "PAUSED"):
                cond = c
                break
        else:
            time.sleep(2)

    if fire_signal:
        # Give the downstream tasks a moment to start so the abort is visible.
        print("Signal is live; letting the downstream tasks start before firing it...")
        time.sleep(10)
        print("Firing the cancel signal...")
        cond.signal(True)
    else:
        print("The pipeline will complete on its own. To cancel it early, run:")
        print(f"    flyte signal condition {r.name} {cond.action_name} true")

    r.wait()
    print("outputs:", r.outputs())
