from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import flyte
from flyte._task import AsyncFunctionTaskTemplate

import trackio

from ._context import (
    _TRACKIO_RUN_KEY,
    clear_trackio_run,
    get_trackio_context,
    set_trackio_run,
)
from ._link import Trackio

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _build_init_kwargs() -> dict[str, Any]:
    """
    Build kwargs for ``trackio.init()`` from the current Flyte context.
    """
    ctx = get_trackio_context()

    if ctx is None:
        return {}

    return ctx.to_trackio_init()


@contextmanager
def _trackio_run(**decorator_kwargs):
    """
    Context manager responsible for Trackio run lifecycle.

    If a parent Trackio run already exists, it is reused.
    Otherwise a new Trackio run is created for the duration
    of the task.
    """

    flyte_ctx = flyte.ctx()

    #
    # Running outside Flyte
    #
    if flyte_ctx is None:
        run = trackio.init(**decorator_kwargs)

        try:
            yield run
        finally:
            run.finish()

        return

    #
    # Reuse parent run if available
    #
    if flyte_ctx.data is None:
        flyte_ctx.data = {}

    saved_run = flyte_ctx.data.get(_TRACKIO_RUN_KEY)

    if saved_run is not None:
        yield saved_run
        return

    context = get_trackio_context()

    init_kwargs = context.to_trackio_init() if context else {}

    init_kwargs.update({k: v for k, v in decorator_kwargs.items() if v is not None})

    run = trackio.init(**init_kwargs)

    set_trackio_run(run)

    try:
        yield run
    finally:
        try:
            run.finish()
        except Exception:
            logger.exception("Failed to finish Trackio run.")
        finally:
            clear_trackio_run()


def trackio_init(
    _func: Optional[F] = None,
    **decorator_kwargs: Any,
) -> F:
    """
    Initialize a Trackio run around a Flyte task.

    Usage
    -----

    @trackio_init
    @env.task
    async def train():
        ...

    @trackio_init(
        project="vision",
        space_id="user/demo",
    )
    @env.task
    async def train():
        ...
    """

    def decorator(task: F) -> F:

        #
        # Flyte Task
        #
        if isinstance(task, AsyncFunctionTaskTemplate):
            #
            # Add Trackio link
            #
            existing_links = getattr(task, "links", ())

            task = task.override(
                links=(
                    *existing_links,
                    Trackio(
                        project=decorator_kwargs.get("project"),
                        server_url=decorator_kwargs.get("server_url"),
                        space_id=decorator_kwargs.get("space_id"),
                    ),
                )
            )

            original_execute = task.execute

            async def wrapped_execute(*args, **kwargs):

                with _trackio_run(**decorator_kwargs):
                    return await original_execute(*args, **kwargs)

            task.execute = wrapped_execute

            return cast(F, task)

        #
        # Plain async Python function
        #
        if iscoroutinefunction(task):

            @functools.wraps(task)
            async def async_wrapper(*args, **kwargs):

                with _trackio_run(**decorator_kwargs):
                    return await task(*args, **kwargs)

            return cast(F, async_wrapper)

        #
        # Plain sync Python function
        #
        @functools.wraps(task)
        def sync_wrapper(*args, **kwargs):

            with _trackio_run(**decorator_kwargs):
                return task(*args, **kwargs)

        return cast(F, sync_wrapper)

    if _func is None:
        return decorator

    return decorator(_func)
