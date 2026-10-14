"""
A run of runs: a driver task launches whole runs with `flyte.run`, waits on them, and calls
`Run.raise_for_status()` to get the same typed exception a sub-action would raise.

The driver first asks for a B300 with `max_queued_time=1m`. On a cluster with no B300s the run
times out in the queue, `raise_for_status()` raises `MaxQueuedTimeExceededError`, and the driver
falls back to a CPU run.

Each child run is wrapped in `@flyte.trace`, and its name is an input to the trace: the driver's own
run name plus a step key, so it is the same on every attempt. If the driver is retried or recovered,
a finished step replays its recorded result (or error). A step that was still waiting when the
driver died runs again under the same name, and creating a run with an existing name returns that
run, so the driver waits on it again instead of launching a duplicate.

    flyte run examples/advanced/run_of_runs.py driver
"""

from datetime import timedelta

import flyte
import flyte.errors

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


async def _run_to_completion(task, run_name: str, **inputs) -> str:
    run = await flyte.with_runcontext(name=run_name).run.aio(task, **inputs)
    print(f"launched {run.name}: {run.url}")
    await run.wait.aio(quiet=True)
    await run.raise_for_status.aio()
    outputs = await run.outputs.aio()
    return outputs[0]


@flyte.trace
async def run_b300(run_name: str, steps: int) -> str:
    return await _run_to_completion(train_on_b300, run_name, steps=steps)


@flyte.trace
async def run_cpu(run_name: str, steps: int) -> str:
    return await _run_to_completion(train_on_cpu, run_name, steps=steps)


@driver_env.task
async def driver(steps: int = 100) -> str:
    # Child run names are stable across attempts of this driver: its run name plus a step key.
    prefix = flyte.ctx().action.run_name
    try:
        return await run_b300(f"{prefix}-b300", steps)
    except flyte.errors.MaxQueuedTimeExceededError as e:
        print(f"no B300 capacity, falling back to CPU: {e}")
        return await run_cpu(f"{prefix}-cpu", steps)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(driver)
    print(run.name)
    print(run.url)
