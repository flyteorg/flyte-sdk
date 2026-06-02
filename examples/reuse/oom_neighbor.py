"""
Demonstrates that an OOM in one task on a reusable replica is observable
by a sibling task running on the same replica.

Usage:
    # Terminal 1 — start the long-running victim first.
    python oom_neighbor.py victim

    # Terminal 2 — within ~30s, start the oomer. Both should land on the
    # same replica (replicas=1, concurrency=5), and the victim should
    # surface an OOM-related failure when the oomer kills the worker.
    python oom_neighbor.py oomer
"""

import asyncio
import sys

import flyte
import flyte.errors

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse")

env = flyte.TaskEnvironment(
    name="oom_neighbor",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    image=actor_image,
    reusable=flyte.ReusePolicy(
        replicas=1,
        concurrency=5,
        idle_ttl=300,
    ),
)


@env.task
async def oomer() -> int:
    # Give the operator time to spin up the victim on the same replica.
    await asyncio.sleep(30)
    big = [0] * 100_000_000
    return len(big)


@env.task(retries=2)
async def victim() -> str:
    try:
        for i in range(60):
            await asyncio.sleep(2)
            print(f"victim alive: {i}")
    except flyte.errors.OOMError as e:
        print(f"victim observed OOM from neighbor: {e}, code={e.code}")
        raise
    return "victim finished without seeing an OOM"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "victim"
    flyte.init_from_config()
    if which == "oomer":
        run = flyte.run(oomer)
    elif which == "victim":
        run = flyte.run(victim)
    else:
        raise SystemExit(f"unknown task {which!r}; expected 'oomer' or 'victim'")
    print(run.url)
    run.wait()
