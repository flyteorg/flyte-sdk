import asyncio
from typing import Any, AsyncGenerator, AsyncIterator, Generic, Iterable, Iterator, List, Union, cast

from flyte.syncify import syncify

from ._group import group
from ._logging import logger
from ._task import P, R, TaskTemplate


class MapAsyncIterator(Generic[P, R]):
    """AsyncIterator implementation for the map function results"""

    def __init__(self, func: TaskTemplate[P, R], args: tuple, name: str, concurrency: int, return_exceptions: bool):
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
            logger.debug(f"Task {self._current_index - 1} failed with exception: {e}")
            if self.return_exceptions:
                return e
            else:
                # Cancel remaining tasks
                for remaining_task in self._tasks[self._current_index + 1 :]:
                    remaining_task.cancel()
                raise e

    async def _initialize(self):
        """Initialize the tasks - called lazily on first iteration"""
        # Create all tasks at once
        tasks = []
        task_count = 0

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

    def __call__(
        self,
        func: TaskTemplate[P, R],
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

        name = self._get_name(func.name, group_name)
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
                    return_exceptions=True,
                ),
            ):
                logger.debug(f"Mapped {x}, task {i}")
                i += 1
                yield x

    async def aio(
        self,
        func: TaskTemplate[P, R],
        *args: Iterable[Any],
        group_name: str | None = None,
        concurrency: int = 0,
        return_exceptions: bool = True,
    ) -> AsyncGenerator[Union[R, Exception], None]:
        if not args:
            return
        name = self._get_name(func.name, group_name)
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
    func: TaskTemplate[P, R],
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
