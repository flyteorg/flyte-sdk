import asyncio
import functools
import logging
from typing import Any, AsyncGenerator, AsyncIterator, Generic, Iterable, Iterator, List, Union, cast, overload

from flyte.syncify import syncify

from ._group import group
from ._logging import logger
from ._task import AsyncFunctionTaskTemplate, F, P, R


class MapAsyncIterator(Generic[P, R]):
    """AsyncIterator implementation for the map function results.

    When ``concurrency > 0`` a bounded worker-pool is used so that only
    *concurrency* asyncio tasks exist at any time - O(concurrency) memory
    regardless of the total number of items.  When ``concurrency == 0`` all
    tasks are created upfront (original behaviour).
    """

    def __init__(
        self,
        func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
        args: tuple,
        name: str,
        concurrency: int,
        return_exceptions: bool,
    ):
        self.func = func
        self.args = args
        self.name = name
        self.concurrency = concurrency
        self.return_exceptions = return_exceptions
        self._current_index = 0
        self._completed_count = 0
        self._exception_count = 0
        self._task_count = 0
        self._initialized = False

        # concurrency == 0 path (all tasks upfront)
        self._tasks: List[asyncio.Task] = []

        # concurrency > 0 path (bounded worker pool)
        self._results: dict[int, tuple[bool, Any]] = {}
        self._condition: asyncio.Condition | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._producer: asyncio.Task | None = None
        self._cancelled = False

    def __aiter__(self) -> AsyncIterator[Union[R, Exception]]:
        return self

    # ------------------------------------------------------------------
    # Invoke helper - handles both plain func and functools.partial
    # ------------------------------------------------------------------
    async def _invoke(self, arg_tuple: tuple) -> R:
        if isinstance(self.func, functools.partial):
            base_func = cast(AsyncFunctionTaskTemplate, self.func.func)
            bound_args = self.func.args
            bound_kwargs = self.func.keywords or {}
            merged_args = bound_args + arg_tuple
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running {base_func.name} with args: {merged_args} and kwargs: {bound_kwargs}")
            return await base_func.aio(*merged_args, **bound_kwargs)
        else:
            return await self.func.aio(*arg_tuple)  # type: ignore[call-overload]

    # ------------------------------------------------------------------
    # __anext__ - dispatches to the right path
    # ------------------------------------------------------------------
    async def __anext__(self) -> Union[R, Exception]:
        if not self._initialized:
            await self._initialize()

        if self._current_index >= self._task_count:
            raise StopAsyncIteration

        idx = self._current_index
        self._current_index += 1

        if self.concurrency > 0:
            return await self._next_bounded(idx)
        else:
            return await self._next_unbounded(idx)

    async def _next_unbounded(self, idx: int) -> Union[R, Exception]:
        """Original path - one pre-created task per item."""
        task = self._tasks[idx]
        try:
            result = await task
            self._completed_count += 1
            return result
        except Exception as e:
            self._exception_count += 1
            if self.return_exceptions:
                return e
            for remaining_task in self._tasks[idx + 1 :]:
                remaining_task.cancel()
            raise e

    async def _next_bounded(self, idx: int) -> Union[R, Exception]:
        """Worker-pool path - wait for the result at *idx*."""
        assert self._condition is not None
        async with self._condition:
            while idx not in self._results:
                await self._condition.wait()
            success, value = self._results.pop(idx)

        if success:
            self._completed_count += 1
            return value
        else:
            self._exception_count += 1
            if self.return_exceptions:
                return value
            # Cancel outstanding work and clean up
            await self._cancel_workers()
            raise value

    async def _cancel_workers(self) -> None:
        """Signal the producer to stop and cancel all in-flight tasks."""
        self._cancelled = True
        if self._producer and not self._producer.done():
            self._producer.cancel()
        for t in list(self._active_tasks):
            t.cancel()
        # Wait briefly for tasks to acknowledge cancellation so they don't
        # leak into the event loop after the iterator is abandoned.
        if self._active_tasks:
            await asyncio.gather(*list(self._active_tasks), return_exceptions=True)
        if self._producer and not self._producer.done():
            await asyncio.gather(self._producer, return_exceptions=True)

    async def aclose(self) -> None:
        """Clean up background tasks if the caller stops iterating early."""
        if self.concurrency > 0 and self._initialized and not self._cancelled:
            await self._cancel_workers()

    # ------------------------------------------------------------------
    # Bounded producer / worker helpers
    # ------------------------------------------------------------------
    async def _produce_bounded(self, arg_tuples: List[tuple]) -> None:
        """Feed work items, blocking when *concurrency* tasks are in-flight."""
        sem = asyncio.Semaphore(self.concurrency)
        for i, at in enumerate(arg_tuples):
            if self._cancelled:
                break
            await sem.acquire()
            if self._cancelled:
                sem.release()
                break
            task = asyncio.create_task(self._run_one(i, at, sem))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    async def _run_one(self, index: int, arg_tuple: tuple, sem: asyncio.Semaphore) -> None:
        assert self._condition is not None
        try:
            result = await self._invoke(arg_tuple)
            async with self._condition:
                self._results[index] = (True, result)
                self._condition.notify_all()
        except Exception as e:
            async with self._condition:
                self._results[index] = (False, e)
                self._condition.notify_all()
        finally:
            sem.release()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    async def _initialize(self) -> None:
        arg_tuples = list(zip(*self.args))
        self._task_count = len(arg_tuples)

        if self._task_count == 0:
            logger.info(f"Group '{self.name}' has no tasks to process")
            self._initialized = True
            return

        if self.concurrency > 0:
            # Bounded path - producer creates at most *concurrency* tasks at a time
            self._condition = asyncio.Condition()
            self._producer = asyncio.create_task(self._produce_bounded(arg_tuples))
            concurrency_desc = str(self.concurrency)
        else:
            # Unbounded path - create all tasks upfront
            self._tasks = [asyncio.create_task(self._invoke(at)) for at in arg_tuples]
            concurrency_desc = "unlimited"

        logger.info(f"Starting {self._task_count} tasks in group '{self.name}' with {concurrency_desc} concurrency")
        self._initialized = True

    async def collect(self) -> List[Union[R, Exception]]:
        results = []
        async for result in self:
            results.append(result)
        return results

    def __repr__(self):
        return f"MapAsyncIterator(group_name='{self.name}', concurrency={self.concurrency})"


class _Mapper(Generic[P, R]):
    """
    Internal mapper class to handle the mapping logic

    NOTE: The reason why we do not use the `@syncify` decorator here is because, in `syncify` we cannot use
    context managers like `group` directly in the function body. This is because the `__exit__` method of the
    context manager is called after the function returns. An for `_context` the `__exit__` method releases the
    token (for contextvar), which was created in a separate thread. This leads to an exception like:

    """

    @classmethod
    def _get_name(cls, task_name: str, group_name: str | None) -> str:
        """Get the name of the group, defaulting to 'map' if not provided."""
        return f"{task_name}_{group_name or 'map'}"

    @staticmethod
    def validate_partial(func: functools.partial[R]):
        """
        This method validates that the provided partial function is valid for mapping, i.e. only the one argument
        is left for mapping and the rest are provided as keywords or args.

        :param func: partial function to validate
        :raises TypeError: if the partial function is not valid for mapping
        """
        f = cast(AsyncFunctionTaskTemplate, func.func)
        inputs = f.native_interface.inputs
        params = list(inputs.keys())
        total_params = len(params)
        provided_args = len(func.args)
        provided_kwargs = len(func.keywords or {})

        # Calculate how many parameters are left unspecified
        unspecified_count = total_params - provided_args - provided_kwargs

        # Exactly one parameter should be left for mapping
        if unspecified_count != 1:
            raise TypeError(
                f"Partial function must leave exactly one parameter unspecified for mapping. "
                f"Found {unspecified_count} unspecified parameters in {f.name}, "
                f"params: {inputs.keys()}"
            )

        # Validate that no parameter is both in args and keywords
        if func.keywords:
            param_names = list(inputs.keys())
            for i, arg_name in enumerate(param_names[: provided_args + 1]):
                if arg_name in func.keywords:
                    raise TypeError(
                        f"Parameter '{arg_name}' is provided both as positional argument and keyword argument "
                        f"in partial function {f.name}."
                    )

    @overload
    def __call__(
        self,
        func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
        *args: Iterable[Any],
        group_name: str | None = None,
        concurrency: int = 0,
    ) -> Iterator[R]: ...

    @overload
    def __call__(
        self,
        func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
        *args: Iterable[Any],
        group_name: str | None = None,
        concurrency: int = 0,
        return_exceptions: bool = True,
    ) -> Iterator[Union[R, Exception]]: ...

    def __call__(
        self,
        func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
        *args: Iterable[Any],
        group_name: str | None = None,
        concurrency: int = 0,
        return_exceptions: bool = True,
    ) -> Iterator[Union[R, Exception]]:
        """
        Map a function over the provided arguments with concurrent execution.

        :param func: The async function to map.
        :param args: Positional arguments to pass to the function (iterables that will be zipped).
        :param group_name: The name of the group for the mapped tasks.
        :param concurrency: The maximum number of concurrent tasks to run. If 0, run all tasks concurrently.
        :param return_exceptions: If True, yield exceptions instead of raising them.
        :return: AsyncIterator yielding results in order.
        """
        if not args:
            return

        if isinstance(func, functools.partial):
            f = cast(AsyncFunctionTaskTemplate, func.func)
            self.validate_partial(func)
        else:
            f = cast(AsyncFunctionTaskTemplate, func)

        name = self._get_name(f.name, group_name)
        logger.debug(f"Blocking Map for {name}")
        with group(name):
            import flyte

            tctx = flyte.ctx()
            if tctx is None or tctx.mode == "local":
                logger.warning("Running map in local mode, which will run every task sequentially.")
                for v in zip(*args):
                    try:
                        yield func(*v)  # type: ignore
                    except Exception as e:
                        if return_exceptions:
                            yield e
                        else:
                            raise e
                return

            i = 0
            for x in cast(
                Iterator[R],
                _map(
                    func,
                    *args,
                    name=name,
                    concurrency=concurrency,
                    return_exceptions=return_exceptions,
                ),
            ):
                logger.debug(f"Mapped {x}, task {i}")
                i += 1
                yield x

    async def aio(
        self,
        func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
        *args: Iterable[Any],
        group_name: str | None = None,
        concurrency: int = 0,
        return_exceptions: bool = True,
    ) -> AsyncGenerator[Union[R, Exception], None]:
        if not args:
            return

        if isinstance(func, functools.partial):
            f = cast(AsyncFunctionTaskTemplate, func.func)
            self.validate_partial(func)
        else:
            f = cast(AsyncFunctionTaskTemplate, func)

        name = self._get_name(f.name, group_name)
        with group(name):
            import flyte

            tctx = flyte.ctx()
            if tctx is None or tctx.mode == "local":
                logger.warning("Running map in local mode, which will run every task sequentially.")
                for v in zip(*args):
                    try:
                        yield func(*v)  # type: ignore
                    except Exception as e:
                        if return_exceptions:
                            yield e
                        else:
                            raise e
                return
            async for x in _map.aio(
                func,
                *args,
                name=name,
                concurrency=concurrency,
                return_exceptions=return_exceptions,
            ):
                yield cast(Union[R, Exception], x)


@syncify
async def _map(
    func: AsyncFunctionTaskTemplate[P, R, F] | functools.partial[R],
    *args: Iterable[Any],
    name: str = "map",
    concurrency: int = 0,
    return_exceptions: bool = True,
) -> AsyncIterator[Union[R, Exception]]:
    iter = MapAsyncIterator(
        func=func, args=args, name=name, concurrency=concurrency, return_exceptions=return_exceptions
    )
    async for result in iter:
        yield result


map: _Mapper = _Mapper()
