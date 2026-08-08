"""simple — the basic task; verify_simple / verify_io / verify_logs each run it."""

from __future__ import annotations

import os

import flyte  # type: ignore

# A stable per-suite suffix keeps each run's TaskEnvironments grouped and lets
# concurrent suites (e.g. different backends) avoid clobbering each other.
_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

_simple_env = flyte.TaskEnvironment(name=f"functional-simple-{_suffix}", cache="disable")


# retries=2: a transient pod/registry blip (ImagePullBackOff, throttled base-image
# pull, CRI hiccup) is retried in-cluster by the backend on a fresh pod.
@_simple_env.task(retries=2)
async def simple(nonce: str) -> str:
    import logging

    logging.getLogger("functional.simple").info(f"simple nonce={nonce}")
    return f"simple-{nonce}"
