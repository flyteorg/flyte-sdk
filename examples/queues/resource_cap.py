"""
Queue resource caps: holding work on the summed request vector, not a count.

`main` submits six 200Mi steps to `caps-mem`, which caps memory at 600Mi and
carries no integer caps. Three run at a time; the rest wait for a predecessor
to release its request.

    flyte run examples/queues/resource_cap.py main

`too_big` submits one 200Mi step to `caps-tiny`, capped at 64Mi. It cannot fit
even an idle queue, so the leasor fails it as UNSCHEDULABLE before dispatch
rather than holding it for room that can never appear. The terminal lands on
the child (`oversized`), not the parent, and no pod is ever created for it.

    flyte run examples/queues/resource_cap.py too_big

Both queues must exist and carry those caps first:

    flyte create queue caps-mem --run-concurrency 100 --action-concurrency 100 \
        --max-resources memory=600Mi
    flyte create queue caps-tiny --run-concurrency 100 --action-concurrency 100 \
        --max-resources memory=64Mi

To watch a cap hold work while the example runs, use the queue dashboard:
`flyte get queue caps-mem --watch`.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial

import flyte

# One environment for both halves: three 200Mi requests fit caps-mem's 600Mi,
# and a single one already exceeds caps-tiny's 64Mi.
env = flyte.TaskEnvironment(
    name="queues_resource_cap",
    resources=flyte.Resources(memory="200Mi"),
)


# 200Mi against a 600Mi cap: three at a time.
@env.task(queue="caps-mem")
async def step(i: int, sleep_seconds: int) -> str:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[caps-mem] step {i} START at {started}", flush=True)
    await asyncio.sleep(sleep_seconds)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[caps-mem] step {i} END   at {finished}", flush=True)
    return f"step {i} done"


@env.task
async def main(count: int = 6, sleep_seconds: int = 4) -> list[str]:
    return list(flyte.map(partial(step, sleep_seconds=sleep_seconds), list(range(count))))


# 200Mi against a 64Mi cap: impossible, not merely blocked. The leasor fails
# this before it is ever dispatched, so the task body never runs.
@env.task(queue="caps-tiny")
async def oversized() -> str:
    return "this should never run"


@env.task
async def too_big() -> str:
    return await oversized()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, count=6, sleep_seconds=4)
    print(run.name)
    print(run.url)
    run.wait()
