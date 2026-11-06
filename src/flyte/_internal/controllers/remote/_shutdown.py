"""Cleanup for event loops owned exclusively by controller threads."""

import asyncio
import sys
import threading

from flyte._logging import logger

# Upper bound on how long stop() waits for controller threads before giving up. These are daemon
# threads, so the process can still exit if they hang. Without the bound, a slow or throttled
# server could keep a finished pod in RUNNING indefinitely.
SHUTDOWN_TIMEOUT_S = 30.0

# Upper bound for each step of draining a loop. Kept well below SHUTDOWN_TIMEOUT_S so the owning
# thread normally finishes before the outer join gives up.
LOOP_DRAIN_STEP_TIMEOUT_S = 10.0


def _run_step(loop: asyncio.AbstractEventLoop, aw, step: str, timeout: float):
    """
    Run `aw` on `loop` for at most `timeout` seconds; returns its result, or None on timeout.

    Uses `asyncio.wait` rather than `wait_for`: on timeout, `wait_for` cancels and then *waits* for the
    awaitable, which never returns if a task swallows cancellation.
    """
    fut = asyncio.ensure_future(aw, loop=loop)
    deadline = loop.time() + timeout
    while not fut.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning(f"Controller loop shutdown: {step} did not finish within {timeout}s, continuing")
            return None
        try:
            loop.run_until_complete(asyncio.wait({fut}, timeout=remaining))
        except RuntimeError:
            # A callback scheduled before shutdown (e.g. a stray loop.stop()) interrupted us; keep waiting.
            continue
    return fut.result()


def drain_and_close_loop(loop: asyncio.AbstractEventLoop, step_timeout: float | None = None) -> None:
    """Run cancellation finalizers before releasing an owned event loop. Each step is time-bounded."""
    if step_timeout is None:
        step_timeout = LOOP_DRAIN_STEP_TIMEOUT_S
    try:
        pending = list(asyncio.all_tasks(loop))
        for task in pending:
            task.cancel()
        if pending:
            results = _run_step(
                loop, asyncio.gather(*pending, return_exceptions=True), "cancelling pending tasks", step_timeout
            )
            for task, result in zip(pending, results or []):
                if isinstance(result, Exception):
                    loop.call_exception_handler(
                        {"message": "Error during controller shutdown", "exception": result, "task": task}
                    )
        _run_step(loop, loop.shutdown_asyncgens(), "async generator shutdown", step_timeout)
        if sys.version_info >= (3, 12):
            # Native timeout: on expiry it warns and stops waiting for the executor threads.
            loop.run_until_complete(loop.shutdown_default_executor(timeout=step_timeout))  # type: ignore[call-arg]
        else:
            _run_step(loop, loop.shutdown_default_executor(), "default executor shutdown", step_timeout)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def join_thread(thread: threading.Thread, timeout: float, what: str) -> bool:
    """
    Wait up to `timeout` seconds for `thread` to exit. Returns True if it exited.

    The timeout is passed to `Thread.join` itself rather than wrapping `to_thread(join)` in
    `wait_for`: the latter would leave an executor worker blocked forever, which in turn hangs
    the caller loop's own default-executor shutdown (e.g. at the end of `asyncio.run`).
    """
    await asyncio.to_thread(thread.join, max(timeout, 0.0))
    if thread.is_alive():
        logger.warning(f"{what} did not exit within {timeout:.1f}s; continuing, it is a daemon thread")
        return False
    return True
