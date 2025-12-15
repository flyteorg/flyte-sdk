import functools
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import wandb

import flyte
from flyte._task import AsyncFunctionTaskTemplate

from .context import get_wandb_context
from .link import Wandb as WandbLink

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def _wandb_run():
    """Context manager for wandb run lifecycle."""
    context_config = get_wandb_context()

    # Convert to wandb.init kwargs
    if context_config:
        config_dict = asdict(context_config)
        # Extract kwargs separately
        extra_kwargs = config_dict.pop("kwargs", None) or {}
        # Merge: explicit fields + extra kwargs (extra_kwargs has lower priority)
        init_kwargs = {
            **extra_kwargs,
            **{k: v for k, v in config_dict.items() if v is not None},
        }
    else:
        init_kwargs = {}

    # Auto-generate ID if not provided
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        init_kwargs["id"] = (
            f"{flyte.ctx().action.run_name}-{flyte.ctx().action.name}"  # TODO: retry attempt and replica index?
        )

    run = wandb.init(**init_kwargs)

    # Store run ID in custom_context
    ctx = flyte.ctx()
    original_custom_context = None
    if ctx and ctx.custom_context is not None:
        original_custom_context = ctx.custom_context.copy()
        ctx.custom_context["_wandb_run_id"] = run.id

    try:
        yield run
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        # Restore original custom_context
        if ctx and original_custom_context is not None:
            ctx.custom_context = original_custom_context


def wandb_init(_func: Optional[F] = None) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and traced functions.

    This decorator:
    1. Initializes a wandb run before execution
    2. Auto-generates unique run ID from Flyte action context (if not provided)
    3. Makes the run available via get_wandb_run()
    4. Automatically finishes the run after completion
    5. For tasks: automatically attaches wandb link (pulls config from context)

    Usage:
        # With Flyte tasks
        @wandb_init
        @env.task
        async def my_task():
            ...

        # With traced functions
        @wandb_init
        @flyte.trace
        async def my_function():
            ...
    """

    def decorator(func: F) -> F:
        # Check if it's a Flyte task (AsyncFunctionTaskTemplate)
        if isinstance(func, AsyncFunctionTaskTemplate):
            # Attach wandb link that pulls from context
            func.link = WandbLink()

            # Wrap the task's execute method with wandb_run
            original_execute = func.execute

            if iscoroutinefunction(original_execute):

                async def wrapped_execute(*args, **kwargs):
                    with _wandb_run():
                        return await original_execute(*args, **kwargs)

                func.execute = wrapped_execute
            else:

                def wrapped_execute(*args, **kwargs):
                    with _wandb_run():
                        return original_execute(*args, **kwargs)

                func.execute = wrapped_execute

            return cast(F, func)
        else:
            # Regular function (e.g. @flyte.trace)
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with _wandb_run():
                        return await func(*args, **kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with _wandb_run():
                        return func(*args, **kwargs)

                return cast(F, sync_wrapper)

    return decorator(_func)


def get_wandb_run():
    """
    Get the current wandb run for this task.

    Reconstructs the run object from the run ID stored in custom_context.
    This allows parent and child tasks to each have their own wandb runs.
    """
    ctx = flyte.ctx()
    if ctx and ctx.custom_context is not None:
        run_id = ctx.custom_context.get("_wandb_run_id")
        if run_id:
            # Reconstruct run from ID
            return wandb.init(id=run_id, resume="allow")
    return None
