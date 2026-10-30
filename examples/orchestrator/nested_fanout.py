"""
Orchestrators that fan out through four levels
==============================================

``root`` calls ``level_2`` n times, each ``level_2`` calls ``level_3`` n
times, each ``level_3`` calls ``level_4`` n times, and each ``level_4`` calls
``leaf`` n times. Every function is a ``flyte.orchestrator.task``, so nothing
is built or deployed first::

    python nested_fanout.py --n 3

The run makes 1 + n + n^2 + n^3 + n^4 actions, and returns the number of
leaves it reached, n^4. ``--n 3`` makes 121 actions; ``--n 5`` makes 781.

In an ``async def`` orchestrator a task call is awaited, as in any async
task, so each level starts all of its children with ``asyncio.gather`` and
they run at once. An orchestrator has to be defined before one that calls
it, because the calls are found when the caller is decorated. The calls pass
their arguments by keyword: the worker does not yet know the order a task
declares its inputs in.
"""

import argparse
import asyncio

import flyte
import flyte.orchestrator


@flyte.orchestrator.task
async def leaf(path: str) -> int:
    return 1


@flyte.orchestrator.task
async def level_4(path: str, n: int) -> int:
    counts = await asyncio.gather(*[leaf(path=path + "." + str(i)) for i in range(n)])
    return sum(counts)


@flyte.orchestrator.task
async def level_3(path: str, n: int) -> int:
    counts = await asyncio.gather(*[level_4(path=path + "." + str(i), n=n) for i in range(n)])
    return sum(counts)


@flyte.orchestrator.task
async def level_2(path: str, n: int) -> int:
    counts = await asyncio.gather(*[level_3(path=path + "." + str(i), n=n) for i in range(n)])
    return sum(counts)


@flyte.orchestrator.task
async def root(n: int) -> int:
    counts = await asyncio.gather(*[level_2(path=str(i), n=n) for i in range(n)])
    return sum(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="children each level fans out to")
    parser.add_argument("--local", action="store_true", help="run in this process instead of on the cluster")
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config()
    run = flyte.run(root, n=args.n)
    print(run.url)
    run.wait()
    print(f"reached {run.outputs()[0]} leaves, expected {args.n**4}")
