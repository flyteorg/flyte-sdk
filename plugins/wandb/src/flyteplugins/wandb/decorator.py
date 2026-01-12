import functools
import logging
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import flyte
from flyte._task import AsyncFunctionTaskTemplate

import wandb

from .context import get_wandb_context, get_wandb_sweep_context
from .link import Wandb, WandbSweep

logger = logging.getLogger(__name__)

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
def _wandb_run(new_run: bool = "auto", func: bool = False, **decorator_kwargs):
    """
    Context manager for wandb run lifecycle.

    Initializes wandb.init() when the context is entered.
    The initialized run is available via flyte.ctx().wandb_run.
    """
    # Try to get Flyte context
    ctx = flyte.ctx()

    # This enables @wandb_init to work in wandb.agent() callbacks (sweep objectives)
    if func and ctx is None:
        # Use config from decorator params (no lazy init for fallback mode)
        run = wandb.init(**decorator_kwargs)
        try:
            yield run
        finally:
            run.finish()
        return
    elif func and ctx:
        raise RuntimeError(
            "@wandb_init cannot be applied to traces. "
            "Traces can access the parent's wandb run via flyte.ctx().wandb_run."
        )

    # Save existing state to restore later
    saved_run_id = ctx.custom_context.get("_wandb_run_id")
    saved_run = ctx.data.get("_wandb_run")

    # Build init kwargs from context
    context_init_kwargs = _build_init_kwargs()
    init_kwargs = {**context_init_kwargs, **decorator_kwargs}

    # Check if this is a trace accessing parent's run
    run = ctx.data.get("_wandb_run")
    if run:
        # This is a trace - yield existing run without initializing
        try:
            yield run
        finally:
            pass  # Don't clean up - parent owns this run
        return

    # Get current action name for run ID generation
    current_action = ctx.action.name

    # Determine if we should reuse parent's run
    should_reuse = False
    if new_run is False:
        should_reuse = True
    elif new_run == "auto":
        should_reuse = bool(saved_run_id)

    # Determine run ID
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        if should_reuse:
            if not saved_run_id:
                raise RuntimeError("Cannot reuse parent run: no parent run ID found")
            init_kwargs["id"] = saved_run_id
        else:
            init_kwargs["id"] = f"{ctx.action.run_name}-{current_action}"

    # Configure reinit parameter (only for local mode)
    # In remote/shared mode, wandb handles run creation/joining automatically
    if flyte.ctx().mode == "local":
        if should_reuse:
            if "reinit" not in init_kwargs:
                init_kwargs["reinit"] = "return_previous"
        else:
            init_kwargs["reinit"] = "create_new"

    # Configure remote mode settings
    if flyte.ctx().mode == "remote":
        is_primary = not should_reuse
        existing_settings = init_kwargs.get("settings", {})

        shared_config = {
            "mode": "shared",
            "x_primary": is_primary,
            "x_label": current_action,
        }
        if not is_primary:
            shared_config["x_update_finish_state"] = False

        init_kwargs["settings"] = wandb.Settings(
            **{**existing_settings, **shared_config}
        )

    # Initialize wandb
    run = wandb.init(**init_kwargs)

    # Store run ID in custom_context (shared with child tasks and accessible to links)
    ctx.custom_context["_wandb_run_id"] = run.id

    # Store run object in ctx.data (task-local only and accessible to traces)
    ctx.data["_wandb_run"] = run

    try:
        yield run
    finally:
        # Determine if this is a primary run
        is_primary_run = new_run is True or (new_run == "auto" and saved_run_id is None)

        if run:
            # Different cleanup logic for local vs remote mode
            should_finish = False

            if flyte.ctx().mode == "remote":
                # In remote/shared mode, always call run.finish() to flush data
                # For secondary tasks, x_update_finish_state=False prevents actually finishing
                # For primary tasks, this properly finishes the run
                should_finish = True
            elif is_primary_run:
                # In local mode, only primary tasks should call run.finish()
                # Secondary tasks reuse the parent's run object, so they must not finish it
                should_finish = True

            if should_finish:
                try:
                    run.finish(exit_code=0)
                except Exception:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                    raise

        # Restore run ID
        if saved_run_id is not None:
            ctx.custom_context["_wandb_run_id"] = saved_run_id
        else:
            ctx.custom_context.pop("_wandb_run_id", None)

        # Restore run object
        if saved_run is not None:
            ctx.data["_wandb_run"] = saved_run
        else:
            ctx.data.pop("_wandb_run", None)


def wandb_init(
    _func: Optional[F] = None,
    *,
    new_run: bool | str = "auto",
    project: Optional[str] = None,
    entity: Optional[str] = None,
    **kwargs,
) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and wandb sweep objectives.

    Args:
        new_run: Controls whether to create a new W&B run or reuse an existing one:
                 - "auto" (default): Creates new run if no parent run exists, otherwise reuses parent's run
                 - True: Always creates a new wandb run with a unique ID
                 - False: Always reuses the parent's run ID (useful for child tasks)
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        **kwargs: Additional wandb.init() parameters (tags, config, mode, etc.)

    Decorator Order:
        For tasks, @wandb_init must be the outermost decorator:
        @wandb_init
        @env.task
        async def my_task():
            ...

    This decorator:
    1. Initializes wandb when the context manager is entered
    2. Auto-generates unique run ID from Flyte action context if not provided
    3. Makes the run available via flyte.ctx().wandb_run
    4. Automatically adds a W&B link to the task in the Flyte UI
    5. Automatically finishes the run after completion
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
            # Create a Wandb link
            # Even if new_run=False, we still add a link - it will point to the parent's run
            wandb_link = Wandb(project=project, entity=entity, new_run=new_run)

            # Get existing links from the task and add wandb link
            existing_links = getattr(func, "links", ())

            # Use override to properly add the link to the task
            func = func.override(links=(*existing_links, wandb_link))

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
        # Regular function
        else:
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(new_run=new_run, func=True, **decorator_kwargs):
                        return await func(*args, **wrapper_kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(new_run=new_run, func=True, **decorator_kwargs):
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
    prior_runs = sweep_config.prior_runs or []

    # Get sweep config dict
    sweep_dict = sweep_config.to_sweep_config()

    # Generate deterministic sweep name if not provided
    if "name" not in sweep_dict or sweep_dict["name"] is None:
        sweep_dict["name"] = f"{ctx.action.run_name}-{ctx.action.name}"

    # Save existing context values to restore later
    saved_sweep_id = ctx.custom_context.get("_wandb_sweep_id")

    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_dict,
        project=project,
        entity=entity,
        prior_runs=prior_runs,
    )

    # Store sweep_id in context (accessible to links)
    ctx.custom_context["_wandb_sweep_id"] = sweep_id

    try:
        yield sweep_id
    finally:
        # Restore previous context values
        if saved_sweep_id is not None:
            ctx.custom_context["_wandb_sweep_id"] = saved_sweep_id
        else:
            ctx.custom_context.pop("_wandb_sweep_id", None)


def wandb_sweep(_func: Optional[F] = None) -> F:
    """
    Decorator to create a wandb sweep and make sweep_id available.

    This decorator:
    1. Creates a wandb sweep using config from context
    2. Makes sweep_id available via flyte.ctx().wandb_sweep_id
    3. Automatically adds a W&B sweep link to the task
    """

    def decorator(func: F) -> F:
        # Check if it's a Flyte task (AsyncFunctionTaskTemplate)
        if isinstance(func, AsyncFunctionTaskTemplate):
            # Create a WandbSweep link
            wandb_sweep_link = WandbSweep()

            # Get existing links from the task and add wandb sweep link
            existing_links = getattr(func, "links", ())

            # Use override to properly add the link to the task
            func = func.override(links=(*existing_links, wandb_sweep_link))

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
            raise RuntimeError("@wandb_sweep can only be used with Flyte tasks.")

    if _func is None:
        return decorator
    return decorator(_func)
