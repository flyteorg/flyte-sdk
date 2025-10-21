import asyncio
from datetime import timedelta

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="long_running",
)


@env.task(report=True)
async def long_running_task(duration: timedelta) -> str:
    """
    A task that simulates a long-running operation.
    It periodically reports a heartbeat to report.log to indicate progress,
    every minute.
    """
    import time

    start_time = time.time()
    end_time = start_time + duration.total_seconds()
    while time.time() < end_time:
        elapsed = time.time() - start_time
        await flyte.report.log.aio(f"<p>Elapsed time: {elapsed:.2f} seconds</p>", do_flush=True)
        await asyncio.sleep(60)


@env.task(report=True)
async def main_task(duration: timedelta) -> str:
    """
    The main task that calls the long-running task.
    """
    await flyte.report.log.aio("<h1>Starting long-running task</h1>", do_flush=True)
    t = asyncio.create_task(long_running_task(duration=duration))
    while not t.done():
        await asyncio.sleep(60)
        await flyte.report.log.aio("<h1>Long-running task still in progress</h1>", do_flush=True)
    return await t


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(main_task, duration=timedelta(days=5))
    print(run.url)
