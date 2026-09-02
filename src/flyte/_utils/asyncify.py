from __future__ import annotations

import asyncio
import contextvars
import inspect
import random
import threading
from concurrent.futures import Future
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from flyte._logging import logger

T = TypeVar("T")
P = ParamSpec("P")


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

    return await asyncio.wrap_future(fut)
