"""
Workloads for the cluster-draining staging regimen (leasor PRs #17847 / #17920 / #17940).

Every task prints the pod hostname and a heartbeat line so leasor-side resets
are visible from the task logs alone: after a force drain the same action shows
up again with a new hostname on another cluster (system retry), while a task
whose queue routes only to the drained cluster never reappears (terminal FAIL).

Pick the queue on the command line; the tasks don't pin one:

    flyte run --queue <queue> examples/cluster_drain/drain_workloads.py <task> [--arg value]
"""

import asyncio
import socket
import time

import flyte

env = flyte.TaskEnvironment(
    name="cluster-drain",
    resources=flyte.Resources(cpu=1, memory="256Mi"),
)


def _where() -> str:
    return socket.gethostname()


@env.task
async def quick() -> str:
    """Placement probe: lands, sleeps 5s, reports where it ran (T0, T1, and after every --activate)."""
    host = _where()
    print(f"quick: running on {host}")
    await asyncio.sleep(5)
    return host


@env.task
async def sleep_for(seconds: int = 1800) -> str:
    """Long runner for T1-T5 and T7. Heartbeats every 10s so a reset shows as a hostname change in the logs."""
    host = _where()
    start = time.monotonic()
    print(f"sleep_for: started on {host}, will run {seconds}s")
    while (elapsed := time.monotonic() - start) < seconds:
        await asyncio.sleep(min(10, seconds - elapsed))
        print(f"sleep_for: {host} alive {int(time.monotonic() - start)}s/{seconds}s")
    print(f"sleep_for: finished on {host}")
    return host


@env.task
async def fan_out(n: int = 50, seconds: int = 300) -> int:
    """
    Parent with n concurrent sleep_for children in the same queue.

    T6: --n 50 --seconds 300, then force-drain and restart the owning leasor shard;
        every child should end on attempt 2, never 3.
    T7: --n 3 --seconds 1800 on the co-named queue; the parent and all children must
        FAIL (not requeue) and the run must end, with no child left Unassigned.
    """
    print(f"fan_out: parent on {_where()}, spawning {n} children x {seconds}s")
    hosts = await asyncio.gather(*(sleep_for(seconds) for _ in range(n)))
    by_host: dict[str, int] = {}
    for h in hosts:
        by_host[h] = by_host.get(h, 0) + 1
    print(f"fan_out: children finished on {by_host}")
    return len(hosts)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(quick)
    print(run.name, run.url)
