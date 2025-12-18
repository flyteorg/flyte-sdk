import functools
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import wandb

import flyte
from flyte._task import AsyncFunctionTaskTemplate

from .context import get_wandb_context, get_wandb_sweep_context

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
def _wandb_run(new_run: bool = True, **decorator_kwargs):
    """
    Context manager for wandb run lifecycle and action marking.

    Works with or without Flyte context:
    - With Flyte context: Full features (parent-child tracking, shared runs, config from context)
    - Without Flyte context: Basic wandb.init() (useful for sweep objectives)
    """
    # Try to get Flyte context
    ctx = flyte.ctx()

    # Fallback mode: No Flyte context available
    # This enables @wandb_init to work in wandb.agent() callbacks (sweep objectives)
    if ctx is None:
        # Use config from decorator params
        run = wandb.init(**decorator_kwargs)
        try:
            yield run
        finally:
            run.finish()
        return

    # Full Flyte-aware mode with enhanced features
    # Save existing state to restore later (task-local)
    saved_run = ctx.data.get("_wandb_run")

    # Save state from custom_context (shared between tasks)
    saved_run_id = ctx.custom_context.get("_wandb_run_id")
    saved_action = ctx.custom_context.get("_wandb_init_action")

    # Mark which action has @wandb_init (shared via custom_context)
    ctx.custom_context["_wandb_init_action"] = ctx.action.name

    # Build init kwargs from context
    init_kwargs = _build_init_kwargs()

    # Merge with decorator params (decorator params take precedence)
    init_kwargs.update(decorator_kwargs)

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

    # Configure shared mode settings - necessary to allow parent-child tasks to log to the same run
    is_primary = new_run or not saved_run_id

    # Get existing settings as dict
    existing_settings = init_kwargs.get("settings", {})

    # Build shared mode configuration
    shared_config = {
        "mode": "shared",
        "x_primary": is_primary,
    }
    if not is_primary:
        shared_config["x_update_finish_state"] = False

    # Merge and create Settings object
    init_kwargs["settings"] = wandb.Settings(**{**existing_settings, **shared_config})

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


def wandb_init(
    _func: Optional[F] = None,
    *,
    new_run: bool = True,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    **kwargs,
) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and traces.

    Works with or without Flyte context - decorator params provide config when context unavailable.

    Args:
        new_run: If True (default), creates a new wandb run with a unique ID.
                 If False, reuses the parent's run ID (useful for child tasks
                 that should log to the same run as their parent).
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        **kwargs: Additional wandb.init() parameters (tags, config, mode, etc.)

    Behavior:
        - Without Flyte context: Uses decorator params directly
        - With Flyte context: Merges context config + decorator params (decorator wins)

    This decorator:
    1. Initializes a wandb run before execution
    2. Auto-generates unique run ID from Flyte action context (if available)
    3. Makes the run available via flyte.ctx().wandb_run (or wandb.run in fallback mode)
    4. Automatically finishes the run after completion
    """

    def decorator(func: F) -> F:
        # Build decorator kwargs dict to pass to _wandb_run
        decorator_kwargs = {}
        if project is not None:
            decorator_kwargs["project"] = project
        if entity is not None:
            decorator_kwargs["entity"] = entity
        decorator_kwargs.update(kwargs)

        # Check if it's a Flyte task (AsyncFunctionTaskTemplate)
        if isinstance(func, AsyncFunctionTaskTemplate):
            # Wrap the task's execute method with wandb_run
            original_execute = func.execute

            if iscoroutinefunction(original_execute):

                async def wrapped_execute(*args, **exec_kwargs):
                    with _wandb_run(new_run=new_run, **decorator_kwargs):
                        return await original_execute(*args, **exec_kwargs)

                func.execute = wrapped_execute
            else:

                def wrapped_execute(*args, **exec_kwargs):
                    with _wandb_run(new_run=new_run, **decorator_kwargs):
                        return original_execute(*args, **exec_kwargs)

                func.execute = wrapped_execute

            return cast(F, func)
        else:
            # Regular function (e.g. @flyte.trace)
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(new_run=new_run, **decorator_kwargs):
                        return await func(*args, **wrapper_kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(new_run=new_run, **decorator_kwargs):
                        return func(*args, **wrapper_kwargs)

                return cast(F, sync_wrapper)

    if _func is None:
        return decorator
    return decorator(_func)


@contextmanager
def _create_sweep():
    """Context manager for wandb sweep creation."""
    ctx = flyte.ctx()

    # Get sweep config from context
    sweep_config = get_wandb_sweep_context()
    if not sweep_config:
        raise RuntimeError(
            "No wandb sweep config found. Use wandb_sweep_config() "
            "with flyte.with_runcontext() or as a context manager."
        )

    # Get wandb config for project/entity (fallback)
    wandb_config = get_wandb_context()
    project = sweep_config.project or (wandb_config.project if wandb_config else None)
    entity = sweep_config.entity or (wandb_config.entity if wandb_config else None)

    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config.to_sweep_config(),
        project=project,
        entity=entity,
    )

    # Store sweep_id in context (string, so custom_context is sufficient)
    ctx.custom_context["_wandb_sweep_id"] = sweep_id

    try:
        yield sweep_id
    finally:
        # Clean up
        ctx.custom_context.pop("_wandb_sweep_id", None)


def wandb_sweep(_func: Optional[F] = None) -> F:
    """
    Decorator to create a wandb sweep and make sweep_id available.

    This decorator:
    1. Creates a wandb sweep using config from context
    2. Makes sweep_id available via flyte.ctx().wandb_sweep_id
    3. Use with wandb.controller() (recommended) or wandb.agent()

    The local controller pattern is recommended for production workloads with Flyte,
    as it allows Flyte to handle orchestration while W&B provides the sweep algorithm.
    """

    def decorator(func: F) -> F:
        # Check if it's a Flyte task (AsyncFunctionTaskTemplate)
        if isinstance(func, AsyncFunctionTaskTemplate):
            original_execute = func.execute

            if iscoroutinefunction(original_execute):

                async def wrapped_execute(*args, **kwargs):
                    with _create_sweep():
                        return await original_execute(*args, **kwargs)

                func.execute = wrapped_execute
            else:

                def wrapped_execute(*args, **kwargs):
                    with _create_sweep():
                        return original_execute(*args, **kwargs)

                func.execute = wrapped_execute

            return cast(F, func)
        else:
            # Regular function (e.g. @flyte.trace)
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with _create_sweep():
                        return await func(*args, **kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with _create_sweep():
                        return func(*args, **kwargs)

                return cast(F, sync_wrapper)

    if _func is None:
        return decorator
    return decorator(_func)
