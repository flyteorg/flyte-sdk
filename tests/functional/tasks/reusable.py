"""Reusable-environment (actor) check — fan out square() over a ReusePolicy env.

Exercises a Union-platform feature (actor reuse); on a backend without reuse
support, skip this scenario.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import flyte  # type: ignore

from . import image_cache_bust

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

_reuse_env = flyte.TaskEnvironment(
    name=f"functional-reuse-{_suffix}",
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse>=0.1.10").with_env_vars(image_cache_bust()),
    # Keep resources small so the reusable actor schedules on modest clusters.
    # concurrency=2 (not 1): reuse_driver itself runs on this env, so with a
    # single slot the driver would hold it and the reuse_square() children it
    # awaits could never get one → self-deadlock. A second slot lets the children
    # run on the same pod, exercising the ReusePolicy path.
    resources=flyte.Resources(memory="256Mi", cpu="250m"),
    cache="disable",
    # The reusable actor re-resolves its env name at pod runtime from its own env,
    # so inject the suffix — else the pod defaults to "smoke" and may look up an
    # unregistered env name.
    env_vars={"FLYTE_FUNCTIONAL_SUFFIX": _suffix},
    reusable=flyte.ReusePolicy(
        replicas=1,
        concurrency=2,
        scaledown_ttl=timedelta(minutes=2),
        idle_ttl=timedelta(minutes=5),
    ),
)


@_reuse_env.task(retries=2)  # tolerate transient pull/scheduling blips on the actor
async def reuse_square(x: int) -> int:
    return x * x


@_reuse_env.task(retries=2)
async def reuse_driver(n: int) -> list[int]:
    """Fan out square() calls over the reusable environment (replicas=1, concurrency=2)."""
    return list(await asyncio.gather(*(reuse_square(i) for i in range(n))))
