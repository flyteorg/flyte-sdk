"""
A parent task fans out to a child that asks for an NVIDIA B300 with a short
``max_queued_time``. B300s are scarce (most clusters have none), so the child's
pod sits Pending, the leaseworker fires ``queued_timeout``, and the child ends
TIMED_OUT.

The interesting part is what the *parent* sees. It catches the exception and
reports its type and message, so you can tell whether the SDK surfaces a
queue-wait timeout distinctly from a runtime timeout.

Run on a real cluster (the budget needs actual pod scheduling to fire):

    flyte --config ~/.flyte/config.yaml run examples/retries_timeout/queued_timeout_b300.py parent
"""

import asyncio
from datetime import timedelta

import flyte
import flyte.errors

b300 = flyte.TaskEnvironment(
    name="b300_queued_timeout_gpu",
    # "B300" is not in the SDK's GPUType literal yet; Device() passes the name through as-is.
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=flyte.Device(device="B300", quantity=1, device_class="GPU")),
)

driver = flyte.TaskEnvironment(
    name="b300_queued_timeout_driver",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    depends_on=[b300],
)


@b300.task(timeout=flyte.Timeout(max_queued_time=timedelta(minutes=1)))
async def train_on_b300() -> str:
    print("train_on_b300: got a B300 (unexpected on a cluster without them)")
    await asyncio.sleep(1)
    return "ran on B300"


@driver.task
async def parent() -> str:
    try:
        return await train_on_b300()
    except flyte.errors.BaseRuntimeError as e:
        report = f"{type(e).__module__}.{type(e).__qualname__} code={e.code!r} kind={e.kind!r}: {e}"
        print(f"parent caught: {report}")
        return report


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(parent)
    print(run.name)
    print(run.url)
