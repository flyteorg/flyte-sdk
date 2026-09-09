"""
Queue resource caps: holding work on the summed request vector, not a count.

`caps-mem` is configured with `maxResources: {memory: 600Mi}` and NO integer
caps, so the only thing that can hold an action is its memory request. Each
step asks for 200Mi, so three fit and the rest wait for a predecessor to
finish — the same shape as small_concurrency.py, measured in memory instead
of actions.

Run it:

    flyte run examples/queues/resource_cap.py main

Expected timing for `count=6, sleep_seconds=4` and 600Mi / 200Mi = 3:
ceil(6/3) * 4s = 8s wall time. Under ~6s means the cap is not being enforced;
far over ~12s means the queue is under-scheduling (a held action should be
re-picked the tick after its predecessor's demand is released, ~1s).

Watch it from the leasor while it runs:

    curl -s localhost:10254/debug/leasor/queues | jq '.queues[]
      | select(.Name=="caps-mem") | {max_resources, in_flight, Depth, ActiveActions}'

    curl -s localhost:10254/metrics | grep -E \
      'queue_resource_(max|in_flight)|schedule_skip_total.*queue_at_resource_cap'

`in_flight.MemoryBytes` should sit at 629145600 (3 x 200Mi) while the queue is
saturated, never above it, and `schedule_skip_total{reason=queue_at_resource_cap}`
should climb while actions are waiting. Both return to zero when the run ends.

---

`too_big` is the other half: one 200Mi step on `caps-tiny`, whose cap is
64Mi. That action cannot fit even an idle queue, so waiting cannot help — the
leasor fails it immediately instead of holding it forever:

    flyte run examples/queues/resource_cap.py too_big

Expect the action to fail in seconds with UNSCHEDULABLE and a message naming
both the request and the cap, and `unschedulable_total{reason="queue_cap"}`
to tick once. It must NOT sit in Unassigned.

Note that it is the CHILD (`oversized`) that carries the UNSCHEDULABLE
terminal — the run fails because its child did, so do not go looking for that
error on the parent. No pod is ever created for the child: it is refused
before dispatch, which is the point.

---

Creating the queues these examples need
---------------------------------------

The wall-time and hold expectations above assume `caps-mem` caps memory at
600Mi and `caps-tiny` at 64Mi. Retune either and the arithmetic changes with
it — the shape to expect is `ceil(count / (cap // request))` waves.

The two queues must exist and carry those caps before either half of this
example demonstrates anything:

    flyte create queue caps-mem --run-concurrency 100 --action-concurrency 100 \
        --max-resources memory=600Mi
    flyte create queue caps-tiny --run-concurrency 100 --action-concurrency 100 \
        --max-resources memory=64Mi

`--max-resources` takes NAME=QUANTITY and repeats per resource, e.g.
`--max-resources gpu=8 --max-resources memory=512Gi`. To retune a cap later,
`flyte update queue caps-mem --edit` shows `max_resources` as a mapping;
setting it to `{}` there removes the cap entirely.

Until a queue actually carries a cap, both examples still run — they simply
demonstrate nothing, because an unconfigured cap admits everything.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial

import flyte

# One environment for everything. The 200Mi request is all both halves need:
# three of them fit caps-mem's 600Mi, and a single one already exceeds
# caps-tiny's 64Mi. A second environment would only add a task the entrypoint's
# deploy does not cover, and a pod k3d may not have room for.
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
