# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
# ]
# ///

"""
Cache serialization demo / regression test.

Exercises ``flyte.Cache(behavior=..., serialize=True)`` — the leaseworker
reservation path that prevents two concurrent executions of the same
``(cache_key, inputs)`` pair from racing. Only one acquires the
``GetOrExtendReservation`` lock; the others observe the reservation is
held, stay in ``ActionPhaseNotStarted`` with ``PluginPhase=Queued``, then
pick up the cached result once the first writes it.

Two scenarios:

  pipeline_serialized   — child uses ``serialize=True``. Fires N concurrent
                          calls with identical inputs. Expectation: exactly
                          one child pod is created; the rest stay queued on
                          the reservation and return the same cached value
                          ~within one heartbeat after the holder finishes.

  pipeline_unserialized — child uses ``serialize=False`` (default). N
                          concurrent calls with identical inputs race —
                          N pods get created and the cache write is
                          last-write-wins. Useful contrast for verifying
                          the ``serialize`` flag actually changes behavior.

Run both:

    flyte run examples/caching/serialized_caching.py pipeline_serialized
    flyte run examples/caching/serialized_caching.py pipeline_unserialized

Verification signals against the local devbox (k3d-devbox, /tmp/devbox.log):

  Serialized (good):
    * Exactly 1 ``Creating Object: ... Kind=pod`` line for child action IDs
    * Multiple ``cache reservation contention: key reservation:... held by
      <holder>, requested by <other>`` log lines
    * Other children finalize within a few seconds of the holder

  Unserialized (contrast):
    * N ``Creating Object: ... Kind=pod`` lines for child action IDs
    * No ``cache reservation contention`` lines
    * All children finalize ~simultaneously after the sleep budget

Verified on test-combined-timeouts (leaseworker-timeouts + implement-timeouts)
2026-05-21: 3-way fan-out produced 1 vs 3 pods respectively and 4 contention
log lines under serialize=True.
"""

import asyncio
import os
import socket
import sys
import time

import flyte
from flyte import Cache

env = flyte.TaskEnvironment(
    "serialized-cache",
    image=flyte.Image.from_uv_script(
        __file__,
        name="serialized-cache",
    ),
)

# The work payload: print enough breadcrumbs so that, if the pod actually
# executes, the leaseworker log shows host + pid + a unique random number.
# When the cache hits, no pod runs and none of these breadcrumbs appear.


@env.task(cache=Cache(behavior="override", version_override="serialized-v1", serialize=True))
async def expensive_child_serialized(seed: int) -> str:
    """Sleeps so the reservation is held long enough for concurrent callers
    to observe it. With serialize=True, only one of N concurrent identical
    callers should ever execute this body."""
    import random

    random.seed()
    marker = random.randint(10_000_000, 99_999_999)
    print(
        f"expensive_child_serialized: EXECUTED seed={seed} marker={marker} "
        f"host={socket.gethostname()} pid={os.getpid()} ts={time.time():.3f}"
    )
    await asyncio.sleep(20)
    return f"seed={seed} marker={marker}"


@env.task(cache=Cache(behavior="override", version_override="unserialized-v1", serialize=False))
async def expensive_child_unserialized(seed: int) -> str:
    """Same body, serialize=False — concurrent callers may all execute."""
    import random

    random.seed()
    marker = random.randint(10_000_000, 99_999_999)
    print(
        f"expensive_child_unserialized: EXECUTED seed={seed} marker={marker} "
        f"host={socket.gethostname()} pid={os.getpid()} ts={time.time():.3f}"
    )
    await asyncio.sleep(20)
    return f"seed={seed} marker={marker}"


@env.task
async def pipeline_serialized(n: int = 3, seed: int = 42) -> list[str]:
    """Fan out N concurrent calls to the serialized child with identical inputs.

    Expected on leaseworker: 1 pod runs, the other N-1 wait on the reservation
    and then return the cached output. All N return values should be identical
    (same `marker` field), proving that exactly one body executed.
    """
    with flyte.group("fanout-serialized"):
        results = await asyncio.gather(*(expensive_child_serialized(seed=seed) for _ in range(n)))
    distinct = sorted(set(results))
    print(f"pipeline_serialized: results={results} distinct_count={len(distinct)}")
    return results


@env.task
async def pipeline_unserialized(n: int = 3, seed: int = 42) -> list[str]:
    """Same fan-out, serialize=False — contrast case."""
    with flyte.group("fanout-unserialized"):
        results = await asyncio.gather(*(expensive_child_unserialized(seed=seed) for _ in range(n)))
    distinct = sorted(set(results))
    print(f"pipeline_unserialized: results={results} distinct_count={len(distinct)}")
    return results


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "pipeline_serialized"
    flyte.init_from_config()
    if scenario == "pipeline_serialized":
        run = flyte.run(pipeline_serialized)
    elif scenario == "pipeline_unserialized":
        run = flyte.run(pipeline_unserialized)
    else:
        print(f"Unknown scenario: {scenario!r}. Use pipeline_serialized or pipeline_unserialized.")
        sys.exit(1)
    print(run.url)
