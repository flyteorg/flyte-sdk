import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import wandb

import flyte
from flyte._task import AsyncFunctionTaskTemplate

from ._context import RankScope, RunMode, get_wandb_context, get_wandb_sweep_context
from ._link import Wandb, WandbSweep

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _get_distributed_info() -> dict | None:
    """
    Auto-detect distributed training info from environment variables.

    Returns None if not in a distributed training context.
    Environment variables are set by torchrun/torch.distributed.elastic.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return None

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    return {
        "rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": world_size,
        "local_world_size": local_world_size,
        "worker_index": int(os.environ.get("GROUP_RANK", "0")),
        "num_workers": world_size // local_world_size if local_world_size > 0 else 1,
    }


def _is_multi_node(info: dict) -> bool:
    """Check if this is a multi-node distributed setup."""
    return info["num_workers"] > 1


def _is_primary_rank(info: dict) -> bool:
    """Check if current process is rank 0 (primary)."""
    return info["rank"] == 0


def _should_skip_rank(
    run_mode: RunMode, rank_scope: RankScope, dist_info: dict
) -> bool:
    """
    Check if this rank should skip wandb initialization.

    For run_mode="auto":
    - rank_scope="global": Only global rank 0 initializes wandb (1 run total)
    - rank_scope="worker": Only local rank 0 of each worker initializes wandb (1 run per worker)

    For run_mode="shared" or "new": All ranks initialize wandb.
    """
    if run_mode != "auto":
        return False

    is_primary = _is_primary_rank(dist_info)
    is_local_primary = dist_info["local_rank"] == 0

    if rank_scope == "global":
        # Global scope: only global rank 0 logs
        return not is_primary
    else:  # rank_scope == "worker"
        # Worker scope: local rank 0 per worker logs
        is_multi_node = _is_multi_node(dist_info)
        if is_multi_node:
            return not is_local_primary
        else:
            return not is_primary


def _configure_distributed_run(
    init_kwargs: dict,
    run_mode: RunMode,
    rank_scope: RankScope,
    dist_info: dict,
    base_run_id: str,
) -> dict:
    """
    Configure wandb.init() kwargs for distributed training.

    Sets run ID, group, and shared mode settings based on:
    - run_mode: "auto", "new", or "shared"
    - rank_scope: "global" or "worker" (affects run ID and grouping)
    - dist_info: distributed topology (rank, worker_index, etc.)
    - base_run_id: base string for generating run IDs

    Run ID patterns:
    - Single-node auto/shared: {base_run_id}
    - Single-node new: {base_run_id}-rank-{rank}
    - Multi-node auto (rank_scope="global"): {base_run_id}
    - Multi-node auto (rank_scope="worker"): {base_run_id}-worker-{worker_index}
    - Multi-node shared (rank_scope="global"): {base_run_id}
    - Multi-node shared (rank_scope="worker"): {base_run_id}-worker-{worker_index}
    - Multi-node new (rank_scope="global"): {base_run_id}-rank-{global_rank}
    - Multi-node new (rank_scope="worker"): {base_run_id}-worker-{worker_index}-rank-{local_rank}
    """
    is_multi_node = _is_multi_node(dist_info)
    is_primary = _is_primary_rank(dist_info)

    # Build run ID based on mode and topology
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        if run_mode == "new":
            # Each rank gets its own run
            if is_multi_node:
                if rank_scope == "global":
                    # Global scope: use global rank for run ID
                    init_kwargs["id"] = f"{base_run_id}-rank-{dist_info['rank']}"
                else:  # rank_scope == "worker"
                    # Worker scope: use worker index and local rank
                    init_kwargs["id"] = (
                        f"{base_run_id}-worker-{dist_info['worker_index']}-rank-{dist_info['local_rank']}"
                    )
            else:
                init_kwargs["id"] = f"{base_run_id}-rank-{dist_info['rank']}"
        elif run_mode == "auto" or run_mode == "shared":
            # For auto and shared mode, run ID depends on rank_scope
            if is_multi_node and rank_scope == "worker":
                # Worker scope: each worker gets or shares its own run
                init_kwargs["id"] = f"{base_run_id}-worker-{dist_info['worker_index']}"
            else:
                # Global scope or single-node: single (shared) run
                init_kwargs["id"] = base_run_id

    # Set group for multiple runs (run_mode="new")
    if run_mode == "new" and "group" not in init_kwargs:
        if is_multi_node and rank_scope == "worker":
            # Worker scope: group per worker
            init_kwargs["group"] = f"{base_run_id}-worker-{dist_info['worker_index']}"
        else:
            # Global scope or single-node: single group for all
            init_kwargs["group"] = base_run_id

    # Configure W&B shared mode for run_mode="shared"
    if run_mode == "shared":
        if is_multi_node:
            x_label = (
                f"worker-{dist_info['worker_index']}-rank-{dist_info['local_rank']}"
            )
            if rank_scope == "global":
                # Global scope: all ranks share one run, only global rank 0 is primary
                is_shared_primary = is_primary  # Only global rank 0
            else:  # rank_scope == "worker"
                # Worker scope: each worker has its own shared run
                is_shared_primary = (
                    dist_info["local_rank"] == 0
                )  # local_rank 0 per worker
        else:
            x_label = f"rank-{dist_info['rank']}"
            # For single-node, primary is rank 0
            is_shared_primary = is_primary

        existing_settings = init_kwargs.get("settings")
        shared_config = {
            "mode": "shared",
            "x_primary": is_shared_primary,
            "x_label": x_label,
            "x_update_finish_state": is_shared_primary,
        }

        # Handle both dict and wandb.Settings objects
        if existing_settings is None:
            init_kwargs["settings"] = wandb.Settings(**shared_config)
        elif isinstance(existing_settings, dict):
            init_kwargs["settings"] = wandb.Settings(**{**existing_settings, **shared_config})
        else:
            # existing_settings is already a wandb.Settings object
            for key, value in shared_config.items():
                setattr(existing_settings, key, value)
            init_kwargs["settings"] = existing_settings

    return init_kwargs


def _build_init_kwargs() -> dict[str, Any]:
    """Build wandb.init() kwargs from current context config."""
    context_config = get_wandb_context()
    if context_config:
        config_dict = asdict(context_config)
        extra_kwargs = config_dict.pop("kwargs", None) or {}

        # Remove Flyte-specific fields that shouldn't be passed to wandb.init()
        config_dict.pop("run_mode", None)
        config_dict.pop("rank_scope", None)
        config_dict.pop("download_logs", None)

        # Filter out None values
        filtered_config = {k: v for k, v in config_dict.items() if v is not None}

        return {**extra_kwargs, **filtered_config}
    return {}


@contextmanager
def _wandb_run(
    run_mode: Optional[RunMode] = None,
    rank_scope: Optional[RankScope] = None,
    func: bool = False,
    **decorator_kwargs,
):
    """
    Context manager for wandb run lifecycle.

    Initializes wandb.init() when the context is entered.
    The initialized run is available via get_wandb_run().
    """
    # Try to get Flyte context
    ctx = flyte.ctx()
    dist_info = _get_distributed_info()

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
        # Check if there's already a W&B run from parent
        saved_run = ctx.data.get("_wandb_run")
        if saved_run:
            yield saved_run
            return

        raise RuntimeError(
            "@wandb_init cannot be applied to traces. Traces can access the parent's wandb run via get_wandb_run()."
        )

    # Save existing state to restore later
    saved_run_id = ctx.custom_context.get("_wandb_run_id")
    saved_run = ctx.data.get("_wandb_run")

    # Build init kwargs from context
    context_init_kwargs = _build_init_kwargs()
    init_kwargs = {**context_init_kwargs, **decorator_kwargs}

    # Get current action name for run ID generation
    current_action = ctx.action.name
    base_run_id = f"{ctx.action.run_name}-{current_action}"

    # Determine effective run_mode and rank_scope: decorator wins if set, otherwise use context
    context_config = get_wandb_context()
    effective_run_mode: RunMode = (
        run_mode or (context_config and context_config.run_mode) or "auto"
    )
    effective_rank_scope: RankScope = (
        rank_scope or (context_config and context_config.rank_scope) or "global"
    )

    # Handle distributed training
    if dist_info:
        if _should_skip_rank(effective_run_mode, effective_rank_scope, dist_info):
            yield None
            return

        init_kwargs = _configure_distributed_run(
            init_kwargs,
            effective_run_mode,
            effective_rank_scope,
            dist_info,
            base_run_id,
        )
    else:
        # Non-distributed training
        # Determine if we should reuse parent's run
        should_reuse = False
        if effective_run_mode == "shared":
            should_reuse = True
        elif effective_run_mode == "auto":
            should_reuse = bool(saved_run_id)

        # Determine run ID
        if "id" not in init_kwargs or init_kwargs["id"] is None:
            if should_reuse:
                if not saved_run_id:
                    raise RuntimeError("Cannot reuse parent run: no parent run ID found")
                init_kwargs["id"] = saved_run_id
            else:
                init_kwargs["id"] = base_run_id

        # Configure reinit parameter (only for local mode)
        if ctx.mode == "local":
            if should_reuse:
                if "reinit" not in init_kwargs:
                    init_kwargs["reinit"] = "return_previous"
            else:
                init_kwargs["reinit"] = "create_new"

        # Configure remote mode settings
        if ctx.mode == "remote":
            is_primary = not should_reuse
            existing_settings = init_kwargs.get("settings")

            shared_config = {
                "mode": "shared",
                "x_primary": is_primary,
                "x_label": current_action,
            }
            if not is_primary:
                shared_config["x_update_finish_state"] = False

            # Handle None, dict, and wandb.Settings objects
            if existing_settings is None:
                init_kwargs["settings"] = wandb.Settings(**shared_config)
            elif isinstance(existing_settings, dict):
                init_kwargs["settings"] = wandb.Settings(**{**existing_settings, **shared_config})
            else:
                # existing_settings is already a wandb.Settings object
                for key, value in shared_config.items():
                    setattr(existing_settings, key, value)
                init_kwargs["settings"] = existing_settings

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
        is_primary_run = effective_run_mode == "new" or (
            effective_run_mode == "auto" and saved_run_id is None
        )

        # Determine if we should call finish()
        should_finish = False
        if run:
            if dist_info and effective_run_mode == "shared":
                # For distributed shared mode, only primary finishes
                if effective_rank_scope == "global":
                    # Global scope: only global rank 0 finishes
                    should_finish = _is_primary_rank(dist_info)
                else:  # rank_scope == "worker"
                    # Worker scope: local_rank 0 of each worker finishes
                    should_finish = dist_info["local_rank"] == 0
            elif (ctx and ctx.mode == "remote") or is_primary_run:
                # In remote mode or for primary runs, always finish
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
    run_mode: Optional[RunMode] = None,
    rank_scope: Optional[RankScope] = None,
    download_logs: Optional[bool] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    **kwargs,
) -> F:
    """
    Decorator to automatically initialize wandb for Flyte tasks and wandb sweep objectives.

    Args:
        run_mode: Controls whether to create a new W&B run or share an existing one:
            - "auto" (default): Creates new run if no parent run exists, otherwise shares parent's run
            - "new": Always creates a new wandb run with a unique ID
            - "shared": Always shares the parent's run ID (useful for child tasks)
            In distributed training context (single-node):
            - "auto" (default): Only rank 0 logs.
            - "shared": All ranks log to a single shared W&B run.
            - "new": Each rank gets its own W&B run (grouped in W&B UI).
            Multi-node: behavior depends on `rank_scope`.
        rank_scope: Flyte-specific rank scope - "global" or "worker".
            Controls which ranks log in distributed training.
            run_mode="auto":
            - "global" (default): Only global rank 0 logs (1 run total).
            - "worker": Local rank 0 of each worker logs (1 run per worker).
            run_mode="shared":
            - "global": All ranks log to a single shared W&B run.
            - "worker": Ranks per worker log to a single shared W&B run (1 run per worker).
            run_mode="new":
            - "global": Each rank gets its own W&B run (1 run total).
            - "worker": Each rank gets its own W&B run grouped per worker -> N runs.
        download_logs: If `True`, downloads wandb run files after task completes
            and shows them as a trace output in the Flyte UI. If None, uses
            the value from `wandb_config()` context if set.
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        **kwargs: Additional `wandb.init()` parameters (tags, config, mode, etc.)

    Decorator Order:
        For tasks, @wandb_init must be the outermost decorator:
        @wandb_init
        @env.task
        async def my_task():
            ...

    This decorator:
    1. Initializes wandb when the context manager is entered
    2. Auto-generates unique run ID from Flyte action context if not provided
    3. Makes the run available via get_wandb_run()
    4. Automatically adds a W&B link to the task in the Flyte UI
    5. Automatically finishes the run after completion
    6. Optionally downloads run logs as a trace output (if download_logs=True)
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
            # Detect distributed config from plugin_config
            nnodes = 1
            nproc_per_node = 1
            plugin_config = getattr(func, "plugin_config", None)

            if plugin_config is not None and type(plugin_config).__name__ == "Elastic":
                nnodes_val = getattr(plugin_config, "nnodes", 1)
                if isinstance(nnodes_val, int):
                    nnodes = nnodes_val
                elif isinstance(nnodes_val, str):
                    parts = nnodes_val.split(":")
                    nnodes = int(parts[-1]) if parts else 1

                nproc_val = getattr(plugin_config, "nproc_per_node", 1)
                if isinstance(nproc_val, int):
                    nproc_per_node = nproc_val
                elif isinstance(nproc_val, str):
                    try:
                        nproc_per_node = int(nproc_val)
                    except ValueError:
                        nproc_per_node = 1

            is_distributed = nnodes > 1 or nproc_per_node > 1

            # Add W&B links
            wandb_id = kwargs.get("id")
            existing_links = getattr(func, "links", ())

            # For links, default to "auto"/"global" if not specified in decorator
            link_run_mode: RunMode = run_mode if run_mode is not None else "auto"
            link_rank_scope: RankScope = (
                rank_scope if rank_scope is not None else "global"
            )

            if nnodes > 1 and link_rank_scope == "worker":
                # Multi-node with worker scope: one link per worker
                wandb_links = tuple(
                    Wandb(
                        project=project,
                        entity=entity,
                        run_mode=link_run_mode,
                        rank_scope=link_rank_scope,
                        id=wandb_id,
                        _is_distributed=True,
                        _worker_index=i,
                        name=f"Weights & Biases Worker {i}",
                    )
                    for i in range(nnodes)
                )
                func = func.override(links=(*existing_links, *wandb_links))
            else:
                # Single-node or multi-node with global scope: one link
                wandb_link = Wandb(
                    project=project,
                    entity=entity,
                    run_mode=link_run_mode,
                    rank_scope=link_rank_scope,
                    id=wandb_id,
                    _is_distributed=is_distributed,
                )
                func = func.override(links=(*existing_links, wandb_link))

            if is_distributed:
                # Distributed: wrap func with sync wrapper
                # The wrapper runs inside each worker after Elastic sets up distributed env vars
                original_fn = func.func

                # Warn if download_logs is requested for distributed tasks
                should_download = download_logs
                if should_download is None:
                    ctx_config = get_wandb_context()
                    should_download = ctx_config.download_logs if ctx_config else False
                if should_download:
                    logger.warning(
                        "download_logs is not supported for distributed tasks. "
                        "Logs will not be downloaded automatically."
                    )

                if iscoroutinefunction(original_fn):
                    logger.warning(
                        "Async task functions are not supported with Elastic. "
                        "Use a sync function instead."
                    )

                    @functools.wraps(original_fn)
                    async def wrapped_fn(*args, **fn_kwargs):
                        with _wandb_run(
                            run_mode=run_mode, rank_scope=rank_scope, **decorator_kwargs
                        ) as run:
                            result = await original_fn(*args, **fn_kwargs)
                        return result

                else:

                    @functools.wraps(original_fn)
                    def wrapped_fn(*args, **fn_kwargs):
                        with _wandb_run(
                            run_mode=run_mode, rank_scope=rank_scope, **decorator_kwargs
                        ) as run:
                            result = original_fn(*args, **fn_kwargs)
                        return result

                func.func = wrapped_fn
            else:
                # Non-distributed: wrap execute with wandb initialization
                original_execute = func.execute

                async def wrapped_execute(*args, **exec_kwargs):
                    with _wandb_run(
                        run_mode=run_mode, rank_scope=rank_scope, **decorator_kwargs
                    ) as run:
                        result = await original_execute(*args, **exec_kwargs)

                    # After run finishes, optionally download logs
                    should_download = download_logs
                    if should_download is None:
                        ctx_config = get_wandb_context()
                        should_download = (
                            ctx_config.download_logs if ctx_config else False
                        )

                    if should_download and run:
                        from . import download_wandb_run_logs

                        await download_wandb_run_logs(run.id)

                    return result

                func.execute = wrapped_execute

            return cast(F, func)
        # Regular function
        else:
            if iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(
                        run_mode=run_mode,
                        rank_scope=rank_scope,
                        func=True,
                        **decorator_kwargs,
                    ):
                        return await func(*args, **wrapper_kwargs)

                return cast(F, async_wrapper)
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **wrapper_kwargs):
                    with _wandb_run(
                        run_mode=run_mode,
                        rank_scope=rank_scope,
                        func=True,
                        **decorator_kwargs,
                    ):
                        return func(*args, **wrapper_kwargs)

                return cast(F, sync_wrapper)

    if _func is None:
        return decorator
    return decorator(_func)


@contextmanager
def _create_sweep(project: Optional[str] = None, entity: Optional[str] = None, **decorator_kwargs):
    """Context manager for wandb sweep creation."""
    ctx = flyte.ctx()

    # Check if a sweep already exists in context - reuse it instead of creating new
    existing_sweep_id = ctx.custom_context.get("_wandb_sweep_id")
    if existing_sweep_id:
        yield existing_sweep_id
        return

    # Get sweep config from context
    sweep_config = get_wandb_sweep_context()
    if not sweep_config:
        raise RuntimeError(
            "No wandb sweep config found. Use wandb_sweep_config() "
            "with flyte.with_runcontext() or as a context manager."
        )

    # Get wandb config for project/entity (fallback)
    wandb_config = get_wandb_context()

    # Priority: decorator kwargs > sweep config > wandb config
    project = project or sweep_config.project or (wandb_config.project if wandb_config else None)
    entity = entity or sweep_config.entity or (wandb_config.entity if wandb_config else None)
    prior_runs = sweep_config.prior_runs or []

    # Get sweep config dict
    sweep_dict = sweep_config.to_sweep_config()

    # Generate deterministic sweep name if not provided
    if "name" not in sweep_dict or sweep_dict["name"] is None:
        sweep_dict["name"] = f"{ctx.action.run_name}-{ctx.action.name}"

    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_dict,
        project=project,
        entity=entity,
        prior_runs=prior_runs,
        **decorator_kwargs,
    )

    # Store sweep_id in context (accessible to links)
    ctx.custom_context["_wandb_sweep_id"] = sweep_id

    try:
        yield sweep_id
    finally:
        # Clean up sweep_id from context
        ctx.custom_context.pop("_wandb_sweep_id", None)


def wandb_sweep(
    _func: Optional[F] = None,
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    download_logs: Optional[bool] = None,
    **kwargs,
) -> F:
    """
    Decorator to create a wandb sweep and make `sweep_id` available.

    This decorator:
    1. Creates a wandb sweep using config from context
    2. Makes `sweep_id` available via `get_wandb_sweep_id()`
    3. Automatically adds a W&B sweep link to the task
    4. Optionally downloads all sweep run logs as a trace output (if `download_logs=True`)

    Args:
        project: W&B project name (overrides context config if provided)
        entity: W&B entity/team name (overrides context config if provided)
        download_logs: if `True`, downloads all sweep run files after task completes
            and shows them as a trace output in the Flyte UI. If None, uses
            the value from wandb_sweep_config() context if set.
        **kwargs: additional `wandb.sweep()` parameters

    Decorator Order:
        For tasks, @wandb_sweep must be the outermost decorator:
        @wandb_sweep
        @env.task
        async def my_task():
            ...
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

            async def wrapped_execute(*args, **exec_kwargs):
                with _create_sweep(project=project, entity=entity, **kwargs) as sweep_id:
                    result = await original_execute(*args, **exec_kwargs)

                # After sweep finishes, optionally download logs
                should_download = download_logs
                if should_download is None:
                    # Check context config
                    sweep_config = get_wandb_sweep_context()
                    should_download = sweep_config.download_logs if sweep_config else False

                if should_download and sweep_id:
                    from . import download_wandb_sweep_logs

                    await download_wandb_sweep_logs(sweep_id)

                return result

            func.execute = wrapped_execute

            return cast(F, func)
        else:
            raise RuntimeError("@wandb_sweep can only be used with Flyte tasks.")

    if _func is None:
        return decorator
    return decorator(_func)
