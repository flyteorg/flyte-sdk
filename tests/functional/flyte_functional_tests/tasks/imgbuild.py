"""Image-builder check — build a custom image (requests) and run a task on it."""

from __future__ import annotations

import os

import flyte  # type: ignore

from . import suite_base_image

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

_imgbuild_env = flyte.TaskEnvironment(
    name=f"functional-imgbuild-{_suffix}",
    image=suite_base_image("requests==2.32.3"),
    cache="disable",
)


@_imgbuild_env.task(retries=2)  # tolerate transient base-image pull / build blips
async def imgbuild_task(nonce: str) -> str:
    import logging

    import requests  # type: ignore

    logging.getLogger("functional.imgbuild").info(f"imgbuild nonce={nonce}")
    return f"requests={requests.__version__}"
