from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

try:
    import trackio
except ImportError as e:
    raise ImportError(
        "The flyteplugins-trackio package requires the 'trackio' package. "
        "Install it with `pip install trackio`."
    ) from e

from .context import (
    clear_trackio_run,
    get_trackio_context,
    merge_trackio_config,
    set_trackio_run,
)

logger = logging.getLogger(__name__)


def _resolve_config(decorator_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve Trackio configuration from the current Flyte context and decorator.

    Configuration precedence:

        decorator kwargs
            >
        trackio_config(...)
            >
        Trackio defaults
    """
    context = get_trackio_context()

    if context is None:
        return decorator_kwargs.copy()

    return merge_trackio_config(
        context.to_dict(),
        decorator_kwargs,
    )


def trackio_init(
    _func: Callable[..., Any] | None = None,
    **decorator_kwargs: Any,
):
    """
    Decorator that initializes a Trackio run for the duration of a Flyte task.

    Examples
    --------

    @trackio_init

    @trackio_init(project="llama")

    @trackio_init(
        project="llama",
        space_id="org/dashboard",
        bucket_id="org/storage",
    )

    All keyword arguments are forwarded directly to ``trackio.init()``.
    """

    def decorator(func: Callable[..., Any]):

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):

                config = _resolve_config(decorator_kwargs)

                run = trackio.init(**config)

                set_trackio_run(run)

                try:
                    return await func(*args, **kwargs)

                finally:
                    try:
                        run.finish()
                    except Exception:
                        logger.exception("Failed to finish Trackio run.")
                    finally:
                        clear_trackio_run()

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            config = _resolve_config(decorator_kwargs)

            run = trackio.init(**config)

            set_trackio_run(run)

            try:
                return func(*args, **kwargs)

            finally:
                try:
                    run.finish()
                except Exception:
                    logger.exception("Failed to finish Trackio run.")
                finally:
                    clear_trackio_run()

        return wrapper

    if _func is None:
        return decorator

    return decorator(_func)