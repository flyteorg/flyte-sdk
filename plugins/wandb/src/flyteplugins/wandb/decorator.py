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

    # Save existing run ID (if any) to restore later
    saved_run_id = (
        ctx.custom_context.get("_wandb_run_id") if ctx and ctx.custom_context else None
    )

    # Build init kwargs from context
    init_kwargs = _build_init_kwargs()

    # Auto-generate ID if not provided
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        init_kwargs["id"] = f"{flyte.ctx().action.run_name}-{flyte.ctx().action.name}"

    run = wandb.init(**init_kwargs)

    # Store this run's ID
    ctx.custom_context["_wandb_run_id"] = run.id

    try:
        yield run
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        # Restore previous run ID
        if ctx and ctx.custom_context is not None:
            if saved_run_id is not None:
                ctx.custom_context["_wandb_run_id"] = saved_run_id
            else:
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
                    ctx = flyte.ctx()
                    # Mark which action has @wandb_init
                    current_action = ctx.action.name
                    saved_action = ctx.custom_context.get("_wandb_init_action")
                    ctx.custom_context["_wandb_init_action"] = current_action

                    try:
                        with _wandb_run():
                            return await original_execute(*args, **kwargs)
                    finally:
                        # Restore previous action marker
                        if saved_action:
                            ctx.custom_context["_wandb_init_action"] = saved_action
                        else:
                            ctx.custom_context.pop("_wandb_init_action", None)

                func.execute = wrapped_execute
            else:

                def wrapped_execute(*args, **kwargs):
                    ctx = flyte.ctx()
                    # Mark which action has @wandb_init
                    current_action = ctx.action.name
                    saved_action = ctx.custom_context.get("_wandb_init_action")
                    ctx.custom_context["_wandb_init_action"] = current_action

                    try:
                        with _wandb_run():
                            return original_execute(*args, **kwargs)
                    finally:
                        # Restore previous action marker
                        if saved_action:
                            ctx.custom_context["_wandb_init_action"] = saved_action
                        else:
                            ctx.custom_context.pop("_wandb_init_action", None)

                func.execute = wrapped_execute

            return cast(F, func)
        else:
            # Regular function (e.g. @flyte.trace)
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    ctx = flyte.ctx()
                    current_action = ctx.action.name
                    saved_action = ctx.custom_context.get("_wandb_init_action")
                    ctx.custom_context["_wandb_init_action"] = current_action

                    try:
                        with _wandb_run():
                            return await func(*args, **kwargs)
                    finally:
                        if saved_action:
                            ctx.custom_context["_wandb_init_action"] = saved_action
                        else:
                            ctx.custom_context.pop("_wandb_init_action", None)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    ctx = flyte.ctx()
                    current_action = ctx.action.name
                    saved_action = ctx.custom_context.get("_wandb_init_action")
                    ctx.custom_context["_wandb_init_action"] = current_action

                    try:
                        with _wandb_run():
                            return func(*args, **kwargs)
                    finally:
                        if saved_action:
                            ctx.custom_context["_wandb_init_action"] = saved_action
                        else:
                            ctx.custom_context.pop("_wandb_init_action", None)

                return cast(F, sync_wrapper)

    return decorator(_func)


def get_wandb_run():
    """
    Get the current wandb run.

    Only returns run if called from within a @wandb_init decorated function.
    Returns None if called from a child function without @wandb_init.
    """
    ctx = flyte.ctx()
    if not ctx or not ctx.custom_context:
        return None

    # Check if current action matches the action that has @wandb_init
    current_action = ctx.action.name
    wandb_action = ctx.custom_context.get("_wandb_init_action")

    if current_action != wandb_action:
        return None  # Called from different action without @wandb_init

    run_id = ctx.custom_context.get("_wandb_run_id")
    if not run_id:
        return None

    return wandb.init(id=run_id, reinit="return_previous")
