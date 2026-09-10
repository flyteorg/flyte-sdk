"""Image-cache check — submit the same stable-image task twice; expect a cache hit."""

from __future__ import annotations

import os

import flyte  # type: ignore

from . import suite_base_image

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

_imgcache_env = flyte.TaskEnvironment(
    name=f"functional-imgcache-{_suffix}",
    image=suite_base_image("requests==2.32.3", extra_env={"FUNCTIONAL_CACHE_TEST": "v1"}),
    cache="disable",
)


@_imgcache_env.task(retries=2)  # tolerate transient base-image pull / build blips
async def imgcache_task(nonce: str) -> str:
    import logging

    import requests  # type: ignore

    logging.getLogger("functional.imgcache").info(f"imgcache nonce={nonce}")
    return f"requests={requests.__version__}"
