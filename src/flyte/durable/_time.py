import asyncio
import time
from datetime import datetime, timezone

import flyte
from flyte.syncify import syncify


@flyte.trace
async def sleep_start() -> float:
    """
    Returns a time at which sleep was started.
    Returns: float
    """
    return time.time()


@syncify
async def durable_sleep(seconds: float):
    """
    durable_sleep enables the process to sleep for `seconds` seconds even if the process recovers from a crash.
    This method can be invoked multiple times. If the process crashes, the invocation of durable_sleep will behave
    like as-if the process has been sleeping since the first time this method was invoked.

    Examples:
    ```python
        import flyte.durable

        env = flyte.TaskEnvironment("env")

        @env.task
        async def main():
            # Do something
            my_work()
            # Now we need to sleep for 1 hour before proceeding.
            await flyte.durable.sleep.aio(3600)  # Even if process crashes, it will resume and only sleep for
                                                  # 1 hour in aggregate. If the scheduling takes longer, it
                                                  # will simply return immediately.
            # thing to be done after 1 hour
            my_work()
    ```

    Args:
        seconds:  float time to sleep for
    """
    start_time = await sleep_start()
    sleep_until = start_time + seconds
    now = time.time()
    # Only if the sleep end time is in the future, sleep until then. If it is in the past then sleep time has elapsed
    # just return
    if sleep_until > now:
        await asyncio.sleep(sleep_until - now)
    return


@syncify
@flyte.trace
async def durable_time() -> float:
    """
    Returns the current time for every unique invocation of durable_time. If the same invocation is encountered again
    the previously returned time is returned again, ensuring determinism.
    Similar to using `time.time()` just durable!
    Returns: float
    """
    return time.time()


@syncify
@flyte.trace
async def durable_now() -> datetime:
    """
    Returns the current time for every unique invocation of durable_now. If the same invocation is encountered
    the previously returned time is returned again, ensuring determinism.
    Similar to using `datetime.now(timezone.utc)` just durable!
    Returns: datetime.datetime, timezone-aware in UTC
    """
    # A replayed value is decoded from a DATETIME literal, which always comes back as an aware UTC datetime.
    # Returning the same form on the first call keeps the result identical across a crash and resume.
    return datetime.now(timezone.utc)
