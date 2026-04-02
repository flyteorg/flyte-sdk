from __future__ import annotations

import asyncio
import contextvars
import inspect
import random
import threading
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from flyte._logging import logger

T = TypeVar("T")
P = ParamSpec("P")


async def run_sync_with_loop(
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """
    Run a synchronous function from an async context with its own event loop.

    This function:
    - Copies the current context variables and preserves them in the sync function
    - Creates a new event loop in a separate thread for the sync function
    - Allows the sync function to potentially use asyncio operations
    - Returns the result without blocking the calling async event loop

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
            result = await run_sync_with_loop(some_sync_function, arg1, arg2)
            return result
    """
    # Check if func is an async function
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            f"Cannot call run_sync_with_loop with async function '{func.__name__}'. "
            "This utility is for running sync functions from async contexts."
        )

    copied_ctx = contextvars.copy_context()
    execute_loop = None
    execute_loop_created = threading.Event()

    # Build thread name with random suffix for uniqueness
    func_name = getattr(func, "__name__", "unknown")
    current_thread = threading.current_thread().name
    random_suffix = f"{random.getrandbits(32):08x}"
    full_thread_name = f"sync-executor-{random_suffix}_from_{current_thread}"

    def _sync_thread_loop_runner() -> None:
        """This method runs the event loop and should be invoked in a separate thread."""
        nonlocal execute_loop
        try:
            execute_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(execute_loop)
            logger.debug(f"Created event loop for function '{func_name}' in thread '{full_thread_name}'")
            execute_loop_created.set()
            execute_loop.run_forever()
        except Exception as e:
            logger.error(f"Exception in thread '{full_thread_name}' running '{func_name}': {e}", exc_info=True)
            raise
        finally:
            if execute_loop:
                logger.debug(f"Stopping event loop for function '{func_name}' in thread '{full_thread_name}'")
                execute_loop.stop()
                execute_loop.close()
                logger.debug(f"Cleaned up event loop for function '{func_name}' in thread '{full_thread_name}'")

    executor_thread = threading.Thread(
        name=full_thread_name,
        daemon=True,
        target=_sync_thread_loop_runner,
    )
    logger.debug(f"Starting executor thread '{full_thread_name}' for function '{func_name}'")
    executor_thread.start()

    async def async_wrapper():
        res = copied_ctx.run(func, *args, **kwargs)
        return res

    # Wait for the loop to be created in a thread to avoid blocking the current thread
    await asyncio.get_event_loop().run_in_executor(None, execute_loop_created.wait)
    assert execute_loop is not None
    fut = asyncio.run_coroutine_threadsafe(async_wrapper(), loop=execute_loop)
    async_fut = asyncio.wrap_future(fut)
    result = await async_fut

    return result
