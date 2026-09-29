"""
A run of runs: a driver task launches whole runs with `flyte.run`, waits on them, and calls
`Run.raise_for_status()` to get the same typed exception a sub-action would raise.

The driver first asks for a B300 with `max_queued_time=1m`. On a cluster with no B300s the run
times out in the queue, `raise_for_status()` raises `MaxQueuedTimeExceededError`, and the driver
falls back to a CPU run.

Only the launch is traced, and it takes the child run's name as an input. The driver derives each
name from its own run name plus a step key, so the name is the same on every attempt. If the driver
is retried or recovered, a recorded launch replays that name; a launch that was in flight when the
driver died runs again with the same name and gets the existing run back. Either way the
driver re-attaches to the same run with `Run.get` instead of launching a duplicate. Waiting and
`raise_for_status()` happen outside the trace, so a replay sees the child run's current state rather
than a recorded result.

    flyte run examples/advanced/run_of_runs.py driver
"""

from datetime import timedelta

import flyte
import flyte.errors
from flyte.remote import Run

gpu = flyte.TaskEnvironment(
    name="run_of_runs_b300",
    # "B300" is not in the SDK's GPUType literal yet; Device() passes the name through as-is.
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=flyte.Device(device="B300", quantity=1, device_class="GPU")),
)

cpu = flyte.TaskEnvironment(name="run_of_runs_cpu", resources=flyte.Resources(cpu=1, memory="1Gi"))

driver_env = flyte.TaskEnvironment(
    name="run_of_runs_driver",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[gpu, cpu],
)


@gpu.task(timeout=flyte.Timeout(max_queued_time=timedelta(minutes=1)))
async def train_on_b300(steps: int) -> str:
    return f"trained {steps} steps on B300"


@cpu.task
async def train_on_cpu(steps: int) -> str:
    return f"trained {steps} steps on CPU"


async def _launch(task, run_name: str, **inputs) -> str:
    try:
        await flyte.with_runcontext(name=run_name).run.aio(task, **inputs)
    except flyte.errors.RuntimeUserError as e:
        # An earlier attempt of this driver launched it and died before the trace was recorded.
        # Creating a run under an existing name returns that run, or raises this error; either
        # way the name refers to the one run.
        if e.code != "RunAlreadyExistsError":
            raise
    return run_name


@flyte.trace
async def launch_b300(run_name: str, steps: int) -> str:
    return await _launch(train_on_b300, run_name, steps=steps)


@flyte.trace
async def launch_cpu(run_name: str, steps: int) -> str:
    return await _launch(train_on_cpu, run_name, steps=steps)


async def wait_for(run_name: str) -> str:
    run = await Run.get.aio(run_name)
    print(f"waiting on {run.name}: {run.url}")
    await run.wait.aio(quiet=True)
    await run.raise_for_status.aio()
    outputs = await run.outputs.aio()
    return outputs[0]


@driver_env.task
async def driver(steps: int = 100) -> str:
    # Child run names are stable across attempts of this driver: its run name plus a step key.
    prefix = flyte.ctx().action.run_name
    try:
        return await wait_for(await launch_b300(f"{prefix}-b300", steps))
    except flyte.errors.MaxQueuedTimeExceededError as e:
        print(f"no B300 capacity, falling back to CPU: {e}")
        return await wait_for(await launch_cpu(f"{prefix}-cpu", steps))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(driver)
    print(run.name)
    print(run.url)
