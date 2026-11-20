"""Remote image builder + image-cache checks."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from . import flyte_ops

pytestmark = pytest.mark.integration


def test_image_builder(flyte_ctx):
    """Build a custom image (requests) and run a task on it."""

    async def _run() -> None:
        from .tasks.imgbuild import imgbuild_task

        nonce = str(uuid.uuid4())
        print(f"[functional] verify_image_builder: submitting (nonce={nonce})", flush=True)
        run = await flyte_ops.submit_with_retry(imgbuild_task, "verify_image_builder", nonce=nonce)
        await flyte_ops.assert_succeeded(run, "verify_image_builder")
        print(f"[functional] verify_image_builder: PASSED (run={run.name})", flush=True)

    asyncio.run(_run())


def test_image_cache(flyte_ctx):
    """Submit the same stable-image task twice; the second run should hit the cache."""

    async def _run() -> None:
        from .tasks.imgcache import imgcache_task

        n1, n2 = str(uuid.uuid4()), str(uuid.uuid4())
        print(f"[functional] verify_image_cache: run 1 (nonce={n1})", flush=True)
        run1 = await flyte_ops.submit_with_retry(imgcache_task, "verify_image_cache/run1", nonce=n1)
        await flyte_ops.assert_succeeded(run1, "verify_image_cache run 1")
        print(f"[functional] verify_image_cache: run 2 (nonce={n2}) — expect cache hit", flush=True)
        run2 = await flyte_ops.submit_with_retry(imgcache_task, "verify_image_cache/run2", nonce=n2)
        await flyte_ops.assert_succeeded(run2, "verify_image_cache run 2")
        print(
            f"[functional] verify_image_cache: PASSED (run1={run1.name} run2={run2.name})",
            flush=True,
        )

    asyncio.run(_run())
