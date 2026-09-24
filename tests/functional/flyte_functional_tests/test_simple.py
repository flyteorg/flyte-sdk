"""Basic task check: submit the simple task and assert it succeeds."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from . import flyte_ops

pytestmark = pytest.mark.integration


def test_simple(flyte_ctx):
    """Submit the simple task and wait for a successful terminal state."""

    async def _run() -> None:
        from .tasks.simple import simple

        nonce = str(uuid.uuid4())
        print(f"[functional] verify_simple: submitting (nonce={nonce})", flush=True)
        run = await flyte_ops.submit_with_retry(simple, "verify_simple", nonce=nonce)
        print(f"[functional] verify_simple: run={run.name}  url={run.url}", flush=True)
        await flyte_ops.assert_succeeded(run, "verify_simple")
        print(f"[functional] verify_simple: PASSED (run={run.name})", flush=True)

    asyncio.run(_run())
