"""
Exercises ``flyte.Timeout(deadline=...)`` — the absolute wall-clock budget
across all attempts of an action, anchored at the first time the action
entered QUEUED.

The task sleeps for 10 minutes; the deadline is 30 seconds. The leasor
should reap the lease and report phase TIMED_OUT (cause: ``deadline``)
well before the sleep finishes.

**Effective deadline note.** The leasor enforces a safety floor of
``LeaseTimeout + HeartbeatInterval`` on the requested deadline — values
below the heartbeat window can't be safely enforced without risking the
leasor reaping a lease before the worker has heartbeated even once. With
default local-server config (``HeartbeatInterval=10s``,
``MaxMissedHeartbeats=3``) the floor is roughly 40-45s, so a requested
``deadline=30s`` will be silently clamped up. You'll still see TIMED_OUT
land well under a minute — just slightly later than the literal 30s the
SDK declared.

To inspect on the server side:

    curl localhost:10254/debug/leasor/state | jq '.run_actions[0] | {action_id, state, info, terminal_result}'
    # rate(leasor:timeouts_total{cause="deadline"}[1m])
"""

import asyncio
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(name="deadline_demo", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task(
    timeout=flyte.Timeout(deadline=timedelta(seconds=30)),
)
async def long_sleeper() -> str:
    """Sleeps 10 minutes — the deadline must reap this well before it returns."""
    print("long_sleeper: starting; will sleep 600s, deadline=30s (clamped to LeaseTimeout+HeartbeatInterval)")
    await asyncio.sleep(600)
    # We should never reach here — the leasor's deadline reap terminates
    # this attempt as TIMED_OUT regardless of remaining retry budget.
    print("long_sleeper: WOKE UP — deadline did NOT fire (unexpected)")
    return "unexpectedly slept through deadline"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(long_sleeper)
    print(run.name)
    print(run.url)
