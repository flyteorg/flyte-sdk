import asyncio
import functools
import logging
from typing import Any, AsyncGenerator, AsyncIterator, Generic, Iterable, Iterator, List, Union, cast, overload

from flyte.syncify import syncify

from ._group import group
from ._logging import logger
from ._task import AsyncFunctionTaskTemplate, F, P, R


class MapAsyncIterator(Generic[P, R]):
    """AsyncIterator implementation for the map function results"""

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
        self._tasks: List[asyncio.Task] = []
        self._current_index = 0
        self._completed_count = 0
        self._exception_count = 0
        self._task_count = 0
        self._initialized = False

    def __aiter__(self) -> AsyncIterator[Union[R, Exception]]:
        """Return self as the async iterator"""
        return self

    async def __anext__(self) -> Union[R, Exception]:
        """Get the next result"""
        # Initialize on first call
        if not self._initialized:
            await self._initialize()

        # Check if we've exhausted all tasks
        if self._current_index >= self._task_count:
            raise StopAsyncIteration

        # Get the next task result
        task = self._tasks[self._current_index]
        self._current_index += 1

        try:
            result = await task
            self._completed_count += 1
            logger.debug(f"Task {self._current_index - 1} completed successfully")
            return result
        except Exception as e:
            self._exception_count += 1
            logger.debug(
                f"Task {self._current_index - 1} failed with exception: {e}, return_exceptions={self.return_exceptions}"
            )
            if self.return_exceptions:
                return e
            else:
                # Cancel remaining tasks
                for remaining_task in self._tasks[self._current_index + 1 :]:
                    remaining_task.cancel()
                logger.warning("Exception raising is `ON`, raising exception and cancelling remaining tasks")
                raise e

    async def _initialize(self):
        """Initialize the tasks - called lazily on first iteration"""
        # Create all tasks at once
        tasks = []
        task_count = 0

        if isinstance(self.func, functools.partial):
            # Handle partial functions by merging bound args/kwargs with mapped args
            base_func = cast(AsyncFunctionTaskTemplate, self.func.func)
            bound_args = self.func.args
            bound_kwargs = self.func.keywords or {}

            for arg_tuple in zip(*self.args):
                # Merge bound positional args with mapped args
                merged_args = bound_args + arg_tuple
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Running {base_func.name} with args: {merged_args} and kwargs: {bound_kwargs}")
                task = asyncio.create_task(base_func.aio(*merged_args, **bound_kwargs))
                tasks.append(task)
                task_count += 1
        else:
            # Handle regular TaskTemplate functions
            for arg_tuple in zip(*self.args):
                task = asyncio.create_task(self.func.aio(*arg_tuple))
                tasks.append(task)
                task_count += 1

        if task_count == 0:
            logger.info(f"Group '{self.name}' has no tasks to process")
            self._tasks = []
            self._task_count = 0
        else:
            logger.info(f"Starting {task_count} tasks in group '{self.name}' with unlimited concurrency")
            self._tasks = tasks
            self._task_count = task_count

        self._initialized = True

    async def collect(self) -> List[Union[R, Exception]]:
        """Convenience method to collect all results into a list"""
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
