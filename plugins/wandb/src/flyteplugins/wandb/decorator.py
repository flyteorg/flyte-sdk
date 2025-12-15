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


def _build_init_kwargs() -> dict[str, Any]:
    """Build wandb.init() kwargs from current context config."""
    context_config = get_wandb_context()
    if context_config:
        config_dict = asdict(context_config)
        extra_kwargs = config_dict.pop("kwargs", None) or {}
        return {
            **extra_kwargs,
            **{k: v for k, v in config_dict.items() if v is not None},
        }
    return {}


@contextmanager
def _wandb_run():
    """Context manager for wandb run lifecycle."""
    ctx = flyte.ctx()

    # Save current run ID (if any)
    saved_run_id = (
        ctx.custom_context.get("_wandb_run_id") if ctx and ctx.custom_context else None
    )

    # Build init kwargs from context
    init_kwargs = _build_init_kwargs()

    # Auto-generate ID if not provided
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        init_kwargs["id"] = (
            f"{flyte.ctx().action.run_name}-{flyte.ctx().action.name}"  # TODO: retry attempt and replica index?
        )

    run = wandb.init(**init_kwargs)

    # Set current run ID and active flag
    if ctx and ctx.custom_context is not None:
        ctx.custom_context["_wandb_run_id"] = run.id
        ctx.custom_context["_wandb_init_active"] = "true"

    try:
        yield run
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        # Remove active flag and restore previous run ID
        if ctx and ctx.custom_context is not None:
            ctx.custom_context.pop("_wandb_init_active", None)
            if saved_run_id is not None:
                ctx.custom_context["_wandb_run_id"] = saved_run_id
            else:
                # No previous run ID, remove the key
                ctx.custom_context.pop("_wandb_run_id", None)


def wandb_init(_func: Optional[F] = None) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and traced functions.

    This decorator:
    1. Initializes a wandb run before execution
    2. Auto-generates unique run ID from Flyte action context (if not provided)
    3. Makes the run available via get_wandb_run()
    4. Automatically finishes the run after completion
    5. For tasks: automatically attaches wandb link (pulls config from context)
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
    Get the current wandb run.

    Only works when called within a @wandb_init decorated task/trace.
    Reconstructs the run object from the run ID stored in custom_context.
    """
    ctx = flyte.ctx()
    if (
        ctx
        and ctx.custom_context is not None
        and "_wandb_init_active" in ctx.custom_context
        and "_wandb_run_id" in ctx.custom_context
    ):
        return wandb.init(
            id=ctx.custom_context["_wandb_run_id"], reinit="return_previous"
        )
    return None
