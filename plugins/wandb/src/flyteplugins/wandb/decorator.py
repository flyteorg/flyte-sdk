import functools
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import wandb

import flyte
from flyte._task import AsyncFunctionTaskTemplate

from .context import get_wandb_context

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
def _wandb_run(new_run: bool = True):
    """Context manager for wandb run lifecycle and action marking."""
    ctx = flyte.ctx()

    # Save existing state to restore later (task-local)
    saved_run = ctx.data.get("_wandb_run")

    # Save state from custom_context (shared between tasks)
    saved_run_id = ctx.custom_context.get("_wandb_run_id")
    saved_action = ctx.custom_context.get("_wandb_init_action")

    # Mark which action has @wandb_init (shared via custom_context)
    ctx.custom_context["_wandb_init_action"] = ctx.action.name

    # Build init kwargs from context
    init_kwargs = _build_init_kwargs()

    # Determine run ID
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        if new_run or not saved_run_id:
            # Create new run ID
            init_kwargs["id"] = f"{ctx.action.run_name}-{ctx.action.name}"
            if "reinit" not in init_kwargs:
                init_kwargs["reinit"] = "create_new"
        else:
            if not saved_run_id:
                raise RuntimeError("Expected saved_run_id when reusing parent's run ID")

            # Reuse parent's run ID
            init_kwargs["id"] = saved_run_id

    run = wandb.init(**init_kwargs)

    # Store run ID in custom_context (shared with child tasks)
    ctx.custom_context["_wandb_run_id"] = run.id

    # Store run object in ctx.data (task-local only)
    ctx.data["_wandb_run"] = run

    try:
        yield run
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        # Restore task-local state
        if saved_run is not None:
            ctx.data["_wandb_run"] = saved_run
        else:
            ctx.data.pop("_wandb_run", None)

        # Restore shared state
        if saved_run_id is not None:
            ctx.custom_context["_wandb_run_id"] = saved_run_id
        else:
            ctx.custom_context.pop("_wandb_run_id", None)

        if saved_action is not None:
            ctx.custom_context["_wandb_init_action"] = saved_action
        else:
            ctx.custom_context.pop("_wandb_init_action", None)


def wandb_init(_func: Optional[F] = None, *, new_run: bool = True) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and traces.

    Args:
        new_run: If True (default), creates a new wandb run with a unique ID.
                 If False, reuses the parent's run ID (useful for child tasks
                 that should log to the same run as their parent).

    This decorator:
    1. Initializes a wandb run before execution
    2. Auto-generates unique run ID from Flyte action context (if not provided)
    3. Makes the run available via flyte.ctx().wandb_run
    4. Automatically finishes the run after completion
    """

    def decorator(func: F) -> F:
        # Check if it's a Flyte task (AsyncFunctionTaskTemplate)
        if isinstance(func, AsyncFunctionTaskTemplate):
            # Wrap the task's execute method with wandb_run
            original_execute = func.execute

            if iscoroutinefunction(original_execute):

                async def wrapped_execute(*args, **kwargs):
                    with _wandb_run(new_run=new_run):
                        return await original_execute(*args, **kwargs)

                func.execute = wrapped_execute
            else:

                def wrapped_execute(*args, **kwargs):
                    with _wandb_run(new_run=new_run):
                        return original_execute(*args, **kwargs)

                func.execute = wrapped_execute

            return cast(F, func)
        else:
            # Regular function (e.g. @flyte.trace)
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with _wandb_run(new_run=new_run):
                        return await func(*args, **kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with _wandb_run(new_run=new_run):
                        return func(*args, **kwargs)

                return cast(F, sync_wrapper)

    if _func is None:
        return decorator
    return decorator(_func)
