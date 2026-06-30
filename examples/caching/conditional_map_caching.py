# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
# ]
# ///

"""
Conditional Map Task Caching

Demonstrates how to cache a parent task's result only when ALL child tasks
in a fan-out succeed, while still allowing the workflow to continue with
partial results when some children fail.

The problem:
    When a parent task fans out work and some children fail, the parent may
    still succeed (by tolerating failures). But if the parent is cached,
    re-running the workflow won't retry the failed children — it will just
    return the cached partial result.

The solution:
    Use a single cached parent task with two call sites:

    1. First call: cached, raises on any child failure (prevents cache write).
    2. Fallback call: same task with ``.override(cache="disable")``, tolerates
       failures and returns partial results so downstream tasks can continue.

    Because each child task is independently cached, the fallback's fan-out
    is cheap — successful children are cache hits and only failed ones re-run.

    On a subsequent re-run (e.g. after transient errors are resolved), the
    first call runs again. If all children now succeed, the parent caches
    and future runs skip the fan-out entirely (single cache lookup).

Scenarios:

    # All children succeed — parent caches on first run
    uv run conditional_map_caching.py all-succeed

    # Some children fail — parent falls back to tolerant mode
    uv run conditional_map_caching.py partial-fail

    # Simulates a re-run after fixing transient errors — previously
    # successful children hit cache, only the fixed ones re-execute,
    # and the parent caches this time
    uv run conditional_map_caching.py rerun-after-fix
"""

import asyncio
import sys

import flyte
from flyte import Cache

env = flyte.TaskEnvironment(
    "conditional-cache",
    image=flyte.Image.from_uv_script(
        __file__,
        name="conditional-cache",
    ),
)


@env.task(cache=Cache(behavior="auto"))
async def child_task(x: int) -> int:
    """Process a single item. Negative values simulate transient failures."""
    if x < 0:
        raise RuntimeError(f"Transient failure processing item {x}")
    return x * x


@env.task(cache=Cache(behavior="auto"))
async def parent_task(items: list[int], tolerate_failures: bool = False) -> list[int | None]:
    """Fan out child_task across all items.

    When ``tolerate_failures`` is False (the default), the task raises after
    all children complete if any of them failed. Because the task raises, its
    result is **not** written to cache — ensuring that a re-run will retry
    the failed children.

    When ``tolerate_failures`` is True, failed children produce ``None`` in
    the output list and the task succeeds. This is used in the fallback call
    with caching disabled so that downstream tasks still receive results.
    """
    with flyte.group("child-fanout"):
        raw = await asyncio.gather(*(child_task(x) for x in items), return_exceptions=True)

    results: list[int | None] = []
    failures = 0
    for r in raw:
        if isinstance(r, Exception):
            failures += 1
            results.append(None)
        else:
            results.append(r)

    if failures > 0 and not tolerate_failures:
        raise RuntimeError(f"{failures}/{len(items)} child tasks failed")

    return results


@env.task
async def downstream_task(results: list[int | None]) -> str:
    """A task that runs after the fan-out, using whatever results are available."""
    succeeded = [r for r in results if r is not None]
    return f"Received {len(succeeded)}/{len(results)} results. Sum = {sum(succeeded)}"


@env.task
async def pipeline(items: list[int]) -> str:
    """Orchestrates the fan-out with conditional caching.

    1. Try the cached parent — raises on any child failure, preventing
       a partial result from being cached.
    2. On failure, fall back to the same parent task with caching disabled
       and failure tolerance enabled. Child tasks that previously succeeded
       are still individually cached, so only the failed ones re-execute.
    3. Pass results downstream regardless of partial failures.
    """
    try:
        results = await parent_task(items=items)
    except Exception as e:
        print(f"Cached parent failed ({e}), retrying with failure tolerance...")
        results = await parent_task.override(cache="disable")(items=items, tolerate_failures=True)

    return await downstream_task(results=results)


SCENARIOS = {
    # All items succeed — parent caches on first run.
    "all-succeed": list(range(20)),
    # Negative values simulate transient failures.
    "partial-fail": [0, 1, 2, -3, 4, 5, -6, 7, 8, 9],
    # Same logical items as partial-fail but with failures resolved.
    # On re-run, children 3 and 6 now succeed. Children 0-2, 4-5, 7-9
    # hit cache from the prior run. Parent caches this time.
    "rerun-after-fix": list(range(10)),
}


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all-succeed"
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario!r}. Choose from: {', '.join(SCENARIOS)}")
        sys.exit(1)

    items = SCENARIOS[scenario]
    print(f"Scenario: {scenario}")
    print(f"Items: {items}")

    flyte.init_from_config()
    run = flyte.run(pipeline, items)
    print(run.url)
