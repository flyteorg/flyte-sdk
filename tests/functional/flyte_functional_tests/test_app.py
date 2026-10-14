"""App-serving check (skipped with --skip-app on backends without app support)."""

from __future__ import annotations

import asyncio

import pytest

from . import flyte_ops

pytestmark = pytest.mark.integration


@pytest.mark.app
def test_app(flyte_ctx):
    """Deploy a FastAPI app, hit its endpoints, and deactivate."""
    asyncio.run(_verify_app(flyte_ctx["suffix"]))


async def _verify_app(suffix: str) -> None:
    from .tasks.app import app_deploy_test

    print("[functional] verify_app: submitting app_deploy_test", flush=True)
    run = await flyte_ops.submit_with_retry(app_deploy_test, "verify_app")
    print(f"[functional] verify_app: run={run.name}  url={run.url}", flush=True)
    # Heaviest scenario (two image builds + a serving cold-start); the default
    # 600s wait is ample for a healthy ~1 min run while still catching a hang.
    try:
        await flyte_ops.assert_succeeded(run, "verify_app")
    except Exception:
        await flyte_ops.dump_app_state(f"functional-app-{suffix}")
        raise
    print(f"[functional] verify_app: PASSED (run={run.name})", flush=True)
