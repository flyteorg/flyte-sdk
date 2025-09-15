import asyncio
import typing


async def run_coros(*coros: typing.Coroutine, return_when: str = asyncio.FIRST_COMPLETED):
    """
    Run a list of coroutines concurrently and wait for the first one to finish or exit.
    When the first one finishes, cancel all other tasks.

    :param coros:
    :param return_when:
    :return:
    """
    # tasks: typing.List[asyncio.Task[typing.Never]] = [asyncio.create_task(c) for c in coros] # Python 3.11+
    tasks: typing.List[asyncio.Task] = [asyncio.create_task(c) for c in coros]
    done, pending = await asyncio.wait(tasks, return_when=return_when)
    # TODO we might want to handle asyncio.CancelledError here, for cases when the `action` is cancelled
    # and we want to propagate it to all tasks. Though the backend will handle it anyway,
    # so this is not strictly necessary.

    for t in pending:  # type: asyncio.Task
        t.cancel()  # Cancel all tasks that didn't finish first

    for t in done:
        err = t.exception()
        if err:
            raise err
