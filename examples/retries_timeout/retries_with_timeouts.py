"""
retries_with_timeouts.py — how ``retries`` (RetryStrategy) interacts with each
timeout bound: ``max_runtime``, ``max_queued_time``, ``deadline``.

A task can declare:

* ``retries=N`` — at most ``N+1`` attempts (the original + N retries).
* ``max_runtime=D`` — wall-clock budget per attempt while the pod is Running.
* ``max_queued_time=D`` — wall-clock budget per attempt while the pod is
  Queued / WaitingForResources / Initializing (i.e., pre-Running).
* ``deadline=D`` — wall-clock budget across *all* attempts, anchored at the
  first time the action was enqueued.

The first two are per-attempt; ``deadline`` is absolute. That asymmetry drives
the retry behavior:

+-----------------+--------------+----------------------------------------------+
| timeout         | retries set? | behavior on a real cluster                   |
+=================+==============+==============================================+
| max_runtime     | yes          | each attempt times out and retries until the |
|                 |              | budget is exhausted (N+1 timed-out attempts) |
+-----------------+--------------+----------------------------------------------+
| max_runtime     | no           | first timeout is final, no retries           |
+-----------------+--------------+----------------------------------------------+
| max_queued_time | yes          | each attempt times out pre-Running and       |
|                 |              | retries until exhausted                      |
+-----------------+--------------+----------------------------------------------+
| max_queued_time | no           | first timeout is final, no retries           |
+-----------------+--------------+----------------------------------------------+
| deadline        | yes          | terminates after ``deadline`` regardless of  |
|                 |              | remaining retries — absolute budget; any     |
|                 |              | still-running downstream children are        |
|                 |              | cascade-aborted by the leasor                |
+-----------------+--------------+----------------------------------------------+
| deadline        | no           | same as above                                |
+-----------------+--------------+----------------------------------------------+

Each task below is intentionally guaranteed to time out so you can compare
attempt counts in the UI. Pick one with ``flyte run``:

    flyte run examples/retries_timeout/retries_with_timeouts.py max_runtime_with_retries
    flyte run examples/retries_timeout/retries_with_timeouts.py queued_timeout_with_retries
    flyte run examples/retries_timeout/retries_with_timeouts.py deadline_with_retries
"""

import asyncio
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(
    name="retries_with_timeouts",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)

# Separate env for the queued-timeout demo. The pod requests an A100 GPU that
# the local devbox cluster cannot satisfy, so it stays Pending and the
# leaseworker's queued_timeout fires deterministically.
env_unschedulable = flyte.TaskEnvironment(
    name="retries_with_queued_timeout",
    resources=flyte.Resources(cpu=1, memory="250Mi", gpu="A100:1"),
)


@env.task(
    retries=2,  # 1 original + 2 retries = up to 3 attempts
    timeout=flyte.Timeout(max_runtime=timedelta(seconds=15)),
)
async def max_runtime_with_retries() -> str:
    """
    Per-attempt ``max_runtime`` budget exceeded on every attempt; retries
    exhaust normally. Expect 3 attempts in the UI, each TIMED_OUT after ~15s.
    Total wall-clock ~= 3 x (pod startup + 15s).
    """
    print("max_runtime_with_retries: will sleep 60s, budget=15s per attempt")
    await asyncio.sleep(60)
    return "unexpected"


@env_unschedulable.task(
    retries=2,
    timeout=flyte.Timeout(max_queued_time=timedelta(seconds=10)),
)
async def queued_timeout_with_retries() -> str:
    """
    Per-attempt ``max_queued_time`` exceeded on every attempt because the pod
    requests an A100 GPU that the local cluster cannot satisfy. Expect 3
    attempts in the UI, each TIMED_OUT after ~10s without ever running the
    user code.
    """
    print("queued_timeout_with_retries: user code reached (unexpected)")
    return "unexpected"


@env.task
async def long_child(idx: int) -> str:
    """A child action that sleeps far longer than the parent's deadline so the
    cascade-abort path has something to actually reap. The leasor's
    handlePendingAbortCascade transitions any still-active children of an
    aborted parent through PendingAbortCascade → PendingAbortCompletion, so
    each of these will surface as ABORTED in the UI once the parent times out.
    """
    print(f"long_child[{idx}]: starting, will sleep 600s")
    await asyncio.sleep(600)
    print(f"long_child[{idx}]: WOKE UP (unexpected — should have been aborted)")
    return f"child {idx} unexpectedly completed"


@env.task(
    retries=5,  # plenty of headroom, but deadline overrides
    timeout=flyte.Timeout(deadline=timedelta(seconds=45)),
)
async def deadline_with_retries() -> list[str]:
    """
    Absolute ``deadline`` budget + cascade-abort to downstream children.

    Spawns three ``long_child`` actions concurrently and waits on them. Each
    child sleeps 600s; the parent's ``deadline=45s`` will fire long before
    any child finishes. Even with ``retries=5`` the deadline is absolute, so
    the parent terminates on its first attempt and the leasor's cascade-abort
    logic walks the parent's child set and transitions every still-active
    child to ABORTED.

    Expected UI shape after the run terminates:

      parent (deadline_with_retries): TIMED_OUT, 1 attempt, ~45s
        ├── long_child[0]: ABORTED (cascade)
        ├── long_child[1]: ABORTED (cascade)
        └── long_child[2]: ABORTED (cascade)

    Total wall-clock ≈ deadline + cascade propagation (typically a second
    or two after the parent terminal).
    """
    print("deadline_with_retries: spawning 3 long-running children; deadline=45s")
    results = await asyncio.gather(
        long_child(idx=0),
        long_child(idx=1),
        long_child(idx=2),
    )
    print("deadline_with_retries: all children returned (unexpected)")
    return list(results)


if __name__ == "__main__":
    # Default to the deadline demo; swap for one of the others to see the
    # corresponding behavior.
    flyte.init_from_config()
    run = flyte.run(deadline_with_retries)
    print(run.name)
    print(run.url)
