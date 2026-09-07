import functools
import inspect
import time
from contextlib import AbstractContextManager, nullcontext
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    TypeGuard,
    TypeVar,
    Union,
    cast,
)

from flyte._logging import logger
from flyte._observe import Recorder, StepInfo, has_observers, observe_step
from flyte.models import NativeInterface, TaskContext
from flyte.syncify import syncify

T = TypeVar("T")


def _step(info: Any, ctx: Any, *, replayed: bool = False) -> StepInfo:
    """Describe this trace step for observers."""
    tctx = ctx.data.task_context
    return StepInfo(
        name=info.name,
        action=info.action,
        task_action=tctx.task_action if tctx else None,
        replayed=replayed,
    )


def _observe(info: Any, ctx: Any) -> AbstractContextManager[Recorder]:
    """Observe this trace step describing it only when something is listening.

    Every traced call reaches this, so the unobserved path must not pay for the observed
    one: without the guard the `StepInfo` would be built as an argument, before
    `observe_step` ever gets the chance to do nothing with it.
    """
    if not has_observers():
        return nullcontext(Recorder())
    return observe_step(_step(info, ctx))


def _observe_replayed(info: Any, ctx: Any) -> None:
    """Record a step that a resumed run served from its durable log instead of running.

    The body is empty on purpose: no work happens on a replay, so the observer sees the
    step and its outcome but no duration. Without this, every step completed before a
    crash would be missing from the resumed run's trace.

    A recorded action is not by itself a replay. The lookup also reports a hit for a step
    whose action exists but holds neither an output nor an error — a traced function with
    no outputs, say — and those still execute, so they get an ordinary span from the
    executing path instead. Only a hit that actually short-circuits execution counts here.
    """
    if not has_observers():
        return
    if info.output is None and info.error is None:
        return
    with observe_step(_step(info, ctx, replayed=True)) as recorder:
        if info.error is not None:
            recorder.record_error(info.error)


@syncify
async def _fetch_action_outputs(controller, iface, func, *args, **kwargs):
    """Module-level proxy: calls controller.get_action_outputs via the global syncify loop."""
    return await controller.get_action_outputs(iface, func, *args, **kwargs)


@syncify
async def _record_trace_action(controller, info):
    """Module-level proxy: calls controller.record_trace via the global syncify loop."""
    await controller.record_trace(info)


def trace(func: Callable[..., T]) -> Callable[..., T]:
    """
    A decorator that traces function execution with timing information.
    Works with regular functions, sync generators, async functions, and async generators/iterators.
    """

    @functools.wraps(func)
    def wrapper_sync(*args: Any, **kwargs: Any) -> Any:
        from flyte._context import Context, internal_ctx

        from ._internal.controllers import get_controller

        ctx = internal_ctx()
        if not ctx.is_task_context():
            return func(*args, **kwargs)

        controller = get_controller()

        # Use syncify only for controller I/O via module-level proxies.
        # `with trace_context` and user code stay on this thread so we do not
        # block syncify's executor across the whole traced call.
        iface = NativeInterface.from_callable(func)
        info, ok = _fetch_action_outputs(controller, iface, func, *args, **kwargs)
        if ok:
            logger.info(f"Found existing trace info for {func}, {info}")
            _observe_replayed(info, ctx)
            if info.output is not None:
                return info.output
            if info.error:
                raise info.error
        else:
            logger.debug(f"No existing trace info found for {func}, proceeding to execute.")

        start_time = time.time()
        trace_task_context = cast(TaskContext, ctx.data.task_context).replace(action=info.action)
        trace_data = ctx.data.replace(task_context=trace_task_context, in_trace=True)
        trace_context = Context(trace_data)

        error = None
        results = None

        with _observe(info, ctx) as recorder, trace_context:
            try:
                results = func(*args, **kwargs)
                info.add_outputs(results, start_time=start_time, end_time=time.time())
            except Exception as e:
                error = e
                recorder.record_error(e)
                info.add_error(e, start_time=start_time, end_time=time.time())

        _record_trace_action(controller, info)
        logger.debug(f"Finished trace for {func}, {info}")

        if error:
            raise error
        return results

    @functools.wraps(func)
    def wrapper_sync_iterator(*args: Any, **kwargs: Any) -> Iterator[Any]:
        from flyte._context import Context, internal_ctx

        from ._internal.controllers import get_controller

        ctx = internal_ctx()
        if not ctx.is_task_context():
            yield from cast(Iterator[Any], func(*args, **kwargs))
            return

        controller = get_controller()

        # Use syncify only for controller I/O via module-level proxies.
        # Keep `with trace_context` and `yield` on this thread to avoid ContextVar
        # token mismatches across threads.
        iface = NativeInterface.from_callable(func)
        info, ok = _fetch_action_outputs(controller, iface, func, *args, **kwargs)
        if ok:
            logger.info(f"Found existing trace info for {func}, {info}")
            _observe_replayed(info, ctx)
            if info.output is not None:
                yield from info.output
                return
            if info.error:
                raise info.error
        else:
            logger.debug(f"No existing trace info found for {func}, proceeding to execute.")

        start_time = time.time()
        trace_task_context = cast(TaskContext, ctx.data.task_context).replace(action=info.action)
        trace_data = ctx.data.replace(task_context=trace_task_context, in_trace=True)
        trace_context = Context(trace_data)

        error = None
        items: list[Any] = []

        with _observe(info, ctx) as recorder, trace_context:
            result = func(*args, **kwargs)
            if inspect.isgenerator(result) or is_sync_iterable(result):
                try:
                    for item in result:
                        items.append(item)
                        yield item
                    info.add_outputs(items, start_time=start_time, end_time=time.time())
                except Exception as e:
                    error = e
                    recorder.record_error(e)
                    info.add_error(e, start_time=start_time, end_time=time.time())

        _record_trace_action(controller, info)
        logger.debug(f"Finished trace for {func}, {info}")

        if error:
            raise error

    @functools.wraps(func)
    async def wrapper_async(*args: Any, **kwargs: Any) -> Any:
        from flyte._context import Context, internal_ctx

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
                _observe_replayed(info, ctx)
                if info.output is not None:
                    return info.output
                elif info.error:
                    raise info.error
            else:
                logger.debug(f"No existing trace info found for {func}, proceeding to execute.")
            start_time = time.time()

            # Create a new context with the trace's action ID and mark as in_trace
            # so that nested task calls run as pure Python instead of submitting to the controller.
            # Note: ctx.data.task_context is guaranteed to be non-None by is_task_context() check above
            trace_task_context = cast(TaskContext, ctx.data.task_context).replace(action=info.action)
            trace_data = ctx.data.replace(task_context=trace_task_context, in_trace=True)
            trace_context = Context(trace_data)

            # Execute function in trace context, then record outside it
            error = None
            results = None

            with _observe(info, ctx) as recorder:
                async with trace_context:
                    # Cast to Awaitable to satisfy mypy
                    coroutine_result = cast(Awaitable[Any], func(*args, **kwargs))
                    try:
                        results = await coroutine_result
                        info.add_outputs(results, start_time=start_time, end_time=time.time())
                    except Exception as e:
                        error = e
                        recorder.record_error(e)
                        info.add_error(e, start_time=start_time, end_time=time.time())

            # Record trace outside the trace context so it uses parent's context
            await controller.record_trace(info)
            logger.debug(f"Finished trace for {func}, {info}")

            if error:
                raise error
            return results
        else:
            # If we are not in a task context, we can just call the function normally
            # Cast to Awaitable to satisfy mypy
            coroutine_result = cast(Awaitable[Any], func(*args, **kwargs))
            return await coroutine_result

    def is_async_iterable(obj: Any) -> TypeGuard[Union[AsyncGenerator, AsyncIterator]]:
        return hasattr(obj, "__aiter__")

    def is_sync_iterable(obj: Any) -> TypeGuard[Iterator[Any]]:
        if isinstance(obj, (str, bytes, bytearray)):
            return False
        if inspect.isasyncgen(obj) or inspect.iscoroutine(obj):
            return False
        return hasattr(obj, "__iter__") and not hasattr(obj, "__aiter__")

    @functools.wraps(func)
    async def wrapper_async_iterator(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        from flyte._context import Context, internal_ctx

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
                _observe_replayed(info, ctx)
                if info.output is not None:
                    for item in info.output:
                        yield item
                elif info.error:
                    raise info.error
            start_time = time.time()

            # Create a new context with the trace's action ID and mark as in_trace
            # so that nested task calls run as pure Python instead of submitting to the controller.
            # Note: ctx.data.task_context is guaranteed to be non-None by is_task_context() check above
            trace_task_context = cast(TaskContext, ctx.data.task_context).replace(action=info.action)
            trace_data = ctx.data.replace(task_context=trace_task_context, in_trace=True)
            trace_context = Context(trace_data)

            # Execute function in trace context, then record outside it
            error = None
            items = []

            with _observe(info, ctx) as recorder:
                async with trace_context:
                    result = func(*args, **kwargs)
                    # TODO ideally we should use streaming into the type-engine so that it stream uploads large blocks
                    if inspect.isasyncgen(result) or is_async_iterable(result):
                        try:
                            # If it's directly an async generator
                            async_iter = result
                            async for item in async_iter:
                                items.append(item)
                                yield item
                            info.add_outputs(items, start_time=start_time, end_time=time.time())
                        except Exception as e:
                            error = e
                            recorder.record_error(e)
                            info.add_error(e, start_time=start_time, end_time=time.time())

            # Record trace outside the trace context so it uses parent's context
            await controller.record_trace(info)

            if error:
                raise error
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
    elif inspect.isgeneratorfunction(func):
        return cast(Callable[..., T], wrapper_sync_iterator)
    else:
        # For regular sync functions
        return cast(Callable[..., T], wrapper_sync)
