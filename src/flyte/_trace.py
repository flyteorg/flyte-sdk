import functools
import inspect
import time
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, TypeGuard, TypeVar, Union, cast

from flyte._logging import logger
from flyte.models import NativeInterface

T = TypeVar("T")


def trace(func: Callable[..., T]) -> Callable[..., T]:
    """
    A decorator that traces function execution with timing information.
    Works with regular functions, async functions, and async generators/iterators.
    """

    @functools.wraps(func)
    def wrapper_sync(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @functools.wraps(func)
    async def wrapper_async(*args: Any, **kwargs: Any) -> Any:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        if ctx.is_task_context():
            # If we are in a task context, that implies we are executing a Run.
            # In this scenario, we should submit the task to the controller.
            # We will also check if we are not initialized, It is not expected to be not initialized
            from ._internal.controllers import get_controller

            controller = get_controller()
            iface = NativeInterface.from_callable(func)
            info, ok = await controller.get_action_outputs(iface, func, *args, **kwargs)
            if ok:
                logger.info(f"Found existing trace info for {func}, {info}")
                if info.output:
                    return info.output
                elif info.error:
                    raise info.error
            else:
                logger.debug(f"No existing trace info found for {func}, proceeding to execute.")
            start_time = time.time()
            try:
                # Cast to Awaitable to satisfy mypy
                coroutine_result = cast(Awaitable[Any], func(*args, **kwargs))
                results = await coroutine_result
                info.add_outputs(results, start_time=start_time, end_time=time.time())
                await controller.record_trace(info)
                logger.debug(f"Finished trace for {func}, {info}")
                return results
            except Exception as e:
                # If there is an error, we need to record it
                info.add_error(e, start_time=start_time, end_time=time.time())
                await controller.record_trace(info)
                raise e
        else:
            # If we are not in a task context, we can just call the function normally
            # Cast to Awaitable to satisfy mypy
            coroutine_result = cast(Awaitable[Any], func(*args, **kwargs))
            return await coroutine_result

    def is_async_iterable(obj: Any) -> TypeGuard[Union[AsyncGenerator, AsyncIterator]]:
        return hasattr(obj, "__aiter__")

    @functools.wraps(func)
    async def wrapper_async_iterator(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        if ctx.is_task_context():
            # If we are in a task context, that implies we are executing a Run.
            # In this scenario, we should submit the task to the controller.
            # We will also check if we are not initialized, It is not expected to be not initialized
            from ._internal.controllers import get_controller

            controller = get_controller()
            iface = NativeInterface.from_callable(func)
            info, ok = await controller.get_action_outputs(iface, func, *args, **kwargs)
            if ok:
                if info.output:
                    for item in info.output:
                        yield item
                elif info.error:
                    raise info.error
            start_time = time.time()
            try:
                items = []
                result = func(*args, **kwargs)
                # TODO ideally we should use streaming into the type-engine so that it stream uploads large blocks
                if inspect.isasyncgen(result) or is_async_iterable(result):
                    # If it's directly an async generator
                    async_iter = result
                    async for item in async_iter:
                        items.append(item)
                        yield item
                info.add_outputs(items, start_time=start_time, end_time=time.time())
                await controller.record_trace(info)
                return
            except Exception as e:
                info.add_error(e, start_time=start_time, end_time=time.time())
                await controller.record_trace(info)
                raise e
        else:
            result = func(*args, **kwargs)
            if is_async_iterable(result):
                async for item in result:
                    yield item

    # Choose the appropriate wrapper based on the function type
    if inspect.iscoroutinefunction(func):
        # This handles async functions that return normal values
        return cast(Callable[..., T], wrapper_async)
    elif inspect.isasyncgenfunction(func):
        return cast(Callable[..., T], wrapper_async_iterator)
    else:
        # For regular sync functions
        return cast(Callable[..., T], wrapper_sync)
