"""Cancel a running pipeline gracefully via a signal (condition).

On startup the pipeline creates a ``cancel`` condition — a signal an external
actor can fire at any time — and races it against its real work. If the signal
fires first, the in-flight child tasks are cancelled (which also aborts their
actions server-side) and the pipeline returns a normal "canceled" result, so
the run still SUCCEEDS.

Contrast with aborting the run, which kills the pod and ends the run ABORTED —
here cancellation is cooperative and nothing is killed.

Run it remotely::

    python cancel_signal.py            # happy path: pipeline completes
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
    """A slow child task — sleeps in 1s steps so there is a window to cancel it."""
    for _ in range(int(work_seconds)):
        await asyncio.sleep(1)
    return f"part {part} done"


@env.task
async def pipeline(work_seconds: float = 30.0) -> str:
    """Create a cancel signal, then race it against two slow child tasks."""
    cancel_signal = await flyte.new_condition.aio(
        "cancel",
        prompt="Fire with `true` to cancel the pipeline gracefully.",
        data_type=bool,
    )
    watcher = asyncio.ensure_future(cancel_signal.wait.aio())
    work = asyncio.gather(process(1, work_seconds), process(2, work_seconds))

    await asyncio.wait({watcher, work}, return_when=asyncio.FIRST_COMPLETED)

    if watcher.done() and watcher.result():
        # Signal fired with `true`: cancel the work, which aborts the child
        # actions server-side too, then exit gracefully.
        work.cancel()
        try:
            await work
        except asyncio.CancelledError:
            pass
        return "canceled: cancel signal was fired"

    watcher.cancel()
    return f"completed: {await work}"


if __name__ == "__main__":
    import sys
    import time

    import flyte.remote as remote

    flyte.init_from_config(log_level="INFO")

    r = flyte.run(pipeline)
    print("run url:", r.url)

    # Wait for the `cancel` condition action to show up.
    cond = None
    while cond is None:
        for c in remote.Condition.listall(run_name=r.name):
            if c.phase.removeprefix("ACTION_PHASE_") in ("RUNNING", "PAUSED"):
                cond = c
                break
        else:
            time.sleep(2)

    if "cancel" in sys.argv[1:]:
        time.sleep(10)  # let the child tasks start so the abort is visible
        print("Firing the cancel signal...")
        cond.signal(True)
    else:
        print("The pipeline will complete on its own. To cancel it early, run:")
        print(f"    flyte signal condition {r.name} {cond.action_name} true")

    r.wait()
    print("outputs:", r.outputs())
