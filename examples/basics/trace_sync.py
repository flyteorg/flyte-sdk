"""
Synchronous @env.task with @flyte.trace on plain (sync) functions.

Each traced step is recorded like an async trace, but the task and helpers use
def + ordinary calls instead of async/await.
"""

from typing import Iterator

import flyte

env = flyte.TaskEnvironment(name="trace_sync_demo", image=flyte.Image.from_debian_base())


@flyte.trace
def double(x: int) -> int:
    return x * 2


@flyte.trace
def range_iterator(x: int) -> Iterator[int]:
    for i in range(x):
        yield i


@env.task
def sum_doubles(n: int) -> int:
    total = 0
    with flyte.group("sum_doubles"):
        for i in range(n):
            total += double(i)

    for i in range_iterator(n):
        total += double(i)

    return total


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(sum_doubles, n=25)
    print(run.url)
