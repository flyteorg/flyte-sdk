"""
Demonstrates the retries + timeout controls covered by the Backoff /
TimeoutStrategy spec. This example is meant to be run **locally**
(``flyte.init()``) so you can observe the local controller honoring the
declared ``Backoff`` policy.

Three things to watch:

1. ``flaky_call`` retries 4 times with exponential backoff (1s, 2s, 4s, capped
   at 3s). The local controller honors the ``Backoff`` policy, so you can see
   the delay grow between attempts in the logs.

2. ``hopeless`` raises ``NonRecoverableError`` and terminates on attempt #1
   even though ``retries=3`` is declared.

3. ``bounded_work`` declares all three timeout bounds (``max_runtime``,
   ``max_queued_time``, ``deadline``). Locally the bounds are recorded on the
   task definition; on a real cluster the leasor and lease worker enforce
   them.

Note: ``flaky_call`` uses a module-level counter to fake flakiness — see the
comment by ``_attempts`` below. That trick only works under the local
controller (single Python process across retries) and is intentional. On a
remote cluster each retry runs in a fresh pod, so module-level state never
carries over and you'd model real flakiness differently.
"""

import asyncio
from datetime import timedelta

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="retry_and_timeout", resources=flyte.Resources(cpu=1, memory="250Mi"))

# LOCAL-ONLY: this module-level counter is how we make `flaky_call` deterministic
# under the local controller — it runs every retry in the same Python process,
# so the global persists across attempts and we can fail twice then succeed.
#
# This will NOT work on a remote cluster: each attempt runs in a fresh pod with
# its own process, so `_attempts` would always start at 0 and the task would
# never converge. The pattern exists purely to exercise the local controller's
# Backoff plumbing in this example. Real flakiness on a cluster comes from the
# environment (network, downstream service), not from in-process state.
_attempts = 0


@env.task(
    retries=flyte.RetryStrategy(
        count=4,
        backoff=flyte.Backoff(
            base=timedelta(seconds=1),
            factor=2.0,
            cap=timedelta(seconds=3),
        ),
    ),
)
async def flaky_call() -> str:
    """
    Fails twice, then succeeds. Retried with exponential backoff.

    LOCAL-ONLY pattern: relies on the module-level ``_attempts`` counter to
    track attempt number across retries. See the comment by ``_attempts``
    above — this works only because the local controller runs every retry
    in the same Python process.
    """
    global _attempts  # noqa: PLW0603
    _attempts += 1
    if _attempts < 3:
        raise RuntimeError(f"transient failure on attempt {_attempts}")
    return f"ok after {_attempts} attempts"


@env.task(retries=3)
async def hopeless(x: int) -> int:
    """``retries=3`` is declared but ``NonRecoverableError`` short-circuits it."""
    raise flyte.errors.NonRecoverableError(f"input {x} can never succeed")


@env.task(
    timeout=flyte.Timeout(
        max_runtime=timedelta(minutes=10),
        max_queued_time=timedelta(minutes=5),
        deadline=timedelta(minutes=30),
    ),
)
async def bounded_work() -> str:
    """All three timeout bounds are declared. Locally this just runs to completion."""
    await asyncio.sleep(0.1)
    return "done"


@env.task
async def main() -> str:
    # Eventually succeeds thanks to retry+backoff.
    result = await flaky_call()

    # Demonstrates non-recoverable error short-circuiting retries.
    try:
        await hopeless(x=-1)
    except flyte.errors.RuntimeUserError as e:
        print(f"hopeless surfaced as: {e.code}")

    # Bounded by all three timeouts; on the cluster the platform would enforce.
    bounded = await bounded_work()
    return f"{result} / {bounded}"


if __name__ == "__main__":
    flyte.init()
    run = flyte.run(main)
    print(run.url)
