from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import random
import threading
from concurrent.futures import Future
from typing import Callable, Iterator, TypeVar

from typing_extensions import ParamSpec

from flyte._logging import logger

T = TypeVar("T")
P = ParamSpec("P")

_wait_for_thread_on_cancel: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "flyte_wait_for_thread_on_cancel", default=False
)


@contextlib.contextmanager
def wait_for_thread_on_cancel() -> Iterator[None]:
    """
    Make a cancelled `run_sync_in_thread` in this context wait for its thread to return before it
    raises `CancelledError`.

    Python can't stop a thread, so a cancelled sync function keeps running. Where many tasks share one
    process (reusable containers), returning on cancel would report the task done and let the runtime start
    the next one in its slot while the old one still runs. Waiting keeps the cancelled task running until
    its thread returns, so the runtime can restart the process if it doesn't.
    """
    token = _wait_for_thread_on_cancel.set(True)
    try:
        yield
    finally:
        _wait_for_thread_on_cancel.reset(token)


async def run_sync_in_thread(
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """
    Run a synchronous function from an async context in a dedicated daemon thread.

    This function:
    - Copies the current context variables into the thread, so the Flyte task context propagates
    - Runs the function as plain sync code: no event loop exists in the thread, so the function
      may freely use `asyncio.run()` or third-party sync wrappers (e.g. an SDK's `run_sync`)
    - Uses a daemon thread, so a stuck function cannot block interpreter exit
    - Returns the result without blocking the calling event loop
    - When cancelled, raises `CancelledError` at once while the thread runs on, or only after the thread
      returns within `wait_for_thread_on_cancel()`

    Args:
        func: The synchronous function to run (must not be an async function)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of calling func(*args, **kwargs)

    Raises:
        TypeError: If func is an async function (coroutine function)

    Example:
        async def my_async_function():
            result = await run_sync_in_thread(some_sync_function, arg1, arg2)
            return result
    """
    # Check if func is an async function
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            f"Cannot call run_sync_in_thread with async function '{getattr(func, '__name__')}'. "
            "This utility is for running sync functions from async contexts."
        )

    copied_ctx = contextvars.copy_context()

    # Build thread name with random suffix for uniqueness
    func_name = getattr(func, "__name__", "unknown")
    current_thread = threading.current_thread().name
    random_suffix = f"{random.getrandbits(32):08x}"
    full_thread_name = f"sync-executor-{random_suffix}_from_{current_thread}"

    fut: Future[T] = Future()

    def _runner() -> None:
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(copied_ctx.run(func, *args, **kwargs))
        except BaseException as e:
            fut.set_exception(e)

    executor_thread = threading.Thread(
        name=full_thread_name,
        daemon=True,
        target=_runner,
    )
    logger.debug(f"Starting executor thread '{full_thread_name}' for function '{func_name}'")
    executor_thread.start()

    result = asyncio.wrap_future(fut)
    if not _wait_for_thread_on_cancel.get():
        return await result
    try:
        return await asyncio.shield(result)
    except asyncio.CancelledError:
        if not result.done():
            logger.warning(
                f"Task cancelled while sync function '{func_name}' is running; Python can't stop a thread, "
                f"so waiting for thread '{full_thread_name}' to return"
            )
        while not result.done():
            try:
                await asyncio.wait([result])
            except asyncio.CancelledError:
                pass
        raise
