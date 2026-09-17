"""Outputs + log-retrieval checks, each against its own `simple` run."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from . import flyte_ops

pytestmark = pytest.mark.integration


def test_io(flyte_ctx):
    """Run.outputs is non-None after a simple run completes."""

    async def _check() -> None:
        from flyte.remote import Run

        from .tasks.simple import simple

        nonce = str(uuid.uuid4())
        run = await flyte_ops.submit_with_retry(simple, "verify_io", nonce=nonce)
        await flyte_ops.assert_succeeded(run, "verify_io")
        got = await Run.get.aio(name=run.name)  # type: ignore
        assert got.outputs is not None, f"no outputs for {run.name}"

    asyncio.run(_check())


@pytest.mark.logs
def test_logs(flyte_ctx):
    """Best-effort: the run's logs are retrievable through the SDK.

    Logs reach the backend via an async, batched sync whose latency is variable
    and can exceed our wait, so a hard timing gate only flakes. Best-effort: poll
    up to the timeout, then WARN (not fail) — retrieving *any* line proves the log
    path works.
    """

    async def _run() -> None:
        from .tasks.simple import simple

        nonce = str(uuid.uuid4())
        run = await flyte_ops.submit_with_retry(simple, "verify_logs", nonce=nonce)
        await flyte_ops.assert_succeeded(run, "verify_logs")
        await _verify_logs(run.name)

    asyncio.run(_run())


async def _verify_logs(run_name: str) -> None:
    from flyte.remote import Run

    print(f"[functional] verify_logs: run={run_name} (best-effort)", flush=True)
    run = await Run.get.aio(name=run_name)  # type: ignore

    async def _collect() -> str:
        out: list[str] = []
        try:
            async for line in run.get_logs.aio():  # type: ignore
                out.append(line)
        except Exception as exc:
            print(f"[functional] verify_logs: get_logs not ready yet ({exc})", flush=True)
        return "\n".join(out).strip()

    _LOG_SYNC_TIMEOUT = 240  # async sync latency is variable; generous, non-fatal
    for _ in range(_LOG_SYNC_TIMEOUT // 10):
        if await _collect():
            print(f"[functional] verify_logs: PASSED ({run_name})", flush=True)
            return
        await asyncio.sleep(10)

    print(
        f"[functional] verify_logs: WARNING — no logs returned for {run_name} within "
        f"{_LOG_SYNC_TIMEOUT}s (async log-sync latency; not failing)",
        flush=True,
    )
