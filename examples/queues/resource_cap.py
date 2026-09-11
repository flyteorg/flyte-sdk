import asyncio
from datetime import datetime, timezone

import flyte

worker = flyte.TaskEnvironment(
    name="queues_caps_worker",
    resources=flyte.Resources(cpu="2", memory="300Mi"),
)

big = flyte.TaskEnvironment(
    name="queues_caps_big",
    resources=flyte.Resources(cpu="8", memory="300Mi"),
)

# The parent declares the environments whose tasks it calls.
driver = flyte.TaskEnvironment(
    name="queues_caps_driver",
    resources=flyte.Resources(memory="200Mi"),
    depends_on=[worker, big],
)


@worker.task(queue="caps-test")
async def hold_two_cpus(i: int, sleep_seconds: int) -> str:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[caps-test] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[caps-test] step {i} END   at {finished}", flush=True)
    return f"held:{i}"


@big.task(queue="caps-test")
async def hold_eight_cpus(sleep_seconds: int) -> str:
    # 8 CPU > the queue's whole 4-CPU cap: can never fit, not merely "not yet".
    await asyncio.sleep(sleep_seconds)
    return "ran (cap not enforced on this deployment)"


@driver.task
async def main(count: int = 3, sleep_seconds: int = 60) -> list[str]:
    return list(
        await asyncio.gather(
            *(hold_two_cpus(i, sleep_seconds) for i in range(count))
        )
    )


@driver.task
async def oversized(sleep_seconds: int = 30) -> str:
    return await hold_eight_cpus(sleep_seconds)
