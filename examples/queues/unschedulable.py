"""
UNSCHEDULABLE fast-fail: a 200Mi task on a queue capped at memory=64Mi
(devbox `caps-tiny`). The demand exceeds the queue's whole cap, so it can
never fit — under a primary `v1-enforced` the lease fails in ~a tick with
code UNSCHEDULABLE (USER / NON_RECOVERABLE, message naming the request and
the cap), no pod is ever created, and unschedulable_total{reason=queue_cap}
ticks once.

Run against a devbox:

    flyte run examples/queues/unschedulable.py main
"""

import asyncio

import flyte

toobig = flyte.TaskEnvironment(
    name="queues_unschedulable",
    resources=flyte.Resources(memory="200Mi"),
)

driver = flyte.TaskEnvironment(
    name="queues_unschedulable_driver",
    resources=flyte.Resources(memory="150Mi"),
    depends_on=[toobig],
)


@toobig.task(queue="caps-tiny")
async def never_fits() -> str:
    await asyncio.sleep(5)
    return "ran (cap not enforced on this deployment)"


@driver.task
async def main() -> str:
    return await never_fits()
