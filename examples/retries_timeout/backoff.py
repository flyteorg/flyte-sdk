"""
backoff.py — exponential backoff between **user** retries.

``RetryStrategy.count`` controls *how many* user retries; ``RetryStrategy.backoff``
controls *how long to wait* between them. The delay before the n-th retry
(0-indexed) is::

    min(base * factor**n, cap)

This example declares ``base=1s, factor=2.0, cap=5s`` over 4 retries, so the
observed gaps grow ``1s → 2s → 4s → 5s`` (the 4th would be 8s but is clamped by
``cap``). Backoff applies to **user** retries only; system retries (network,
container, k8s) are paced by the platform, not by this policy.

Run it **locally** so the local controller actually sleeps between attempts and
you can watch the gaps grow::

    python examples/retries_timeout/backoff.py

The controller logs ``retrying in X.XXs...`` between attempts; the task itself
also prints the wall-clock gap it observed, so the backoff curve is visible from
both sides.

LOCAL-ONLY caveat: the ``_attempts`` / ``_last_ts`` module globals below survive
across retries only because the local controller runs every attempt in the same
Python process. On a real cluster each attempt is a fresh pod, so this in-process
state never carries over — model real flakiness from the environment instead.
"""

import asyncio
import time
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(name="backoff_demo", resources=flyte.Resources(cpu=1, memory="250Mi"))

# LOCAL-ONLY in-process state — see the module docstring.
_attempts = 0
_last_ts: float | None = None


@env.task(
    retries=flyte.RetryStrategy(
        count=4,  # 1 original attempt + 4 retries = up to 5 attempts
        backoff=flyte.Backoff(
            base=timedelta(seconds=1),  # delay before retry #0
            factor=10.0,  # double each retry: 1s, 2s, 4s, 8s...
            cap=timedelta(seconds=30),  # ...clamped to 5s, so 4th gap is 5s not 8s
        ),
    ),
)
async def flaky() -> str:
    """
    Fails the first 4 attempts, succeeds on the 5th. The interesting part is the
    *spacing* between attempts, not the eventual success — watch the printed gaps
    grow 1s → 2s → 4s → 5s (capped).
    """
    global _attempts, _last_ts  # noqa: PLW0603
    _attempts += 1

    now = time.monotonic()
    gap = "(first attempt)" if _last_ts is None else f"{now - _last_ts:.2f}s since previous attempt"
    _last_ts = now
    print(f"flaky: attempt {_attempts} — {gap}")

    if _attempts < 5:
        raise RuntimeError(f"transient failure on attempt {_attempts}")
    return f"ok after {_attempts} attempts"


@env.task
async def main() -> str:
    result = await flaky()
    print(f"main: {result}")
    return result


if __name__ == "__main__":
    # Local execution: the local controller honors the Backoff policy, sleeping
    # the computed delay between attempts so the growing gaps are observable.
    flyte.init()
    run = flyte.run(main)
    print(run.url)
