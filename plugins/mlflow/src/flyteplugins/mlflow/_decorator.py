import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import flyte
from flyte._task import AsyncFunctionTaskTemplate

import mlflow

from ._context import RunMode, get_mlflow_context
from ._link import Mlflow

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Fields handled by _setup_tracking (not passed to mlflow.start_run())
_TRACKING_FIELDS = {"tracking_uri", "experiment_name", "experiment_id"}

# Plugin-specific fields (not passed to mlflow.start_run())
_PLUGIN_FIELDS = {
    "run_mode",
    "autolog",
    "framework",
    "log_models",
    "log_datasets",
    "autolog_kwargs",
    "link_host",
    "link_template",
}

_DEFAULT_LINK_TEMPLATE = "{host}/#/experiments/{experiment_id}/runs/{run_id}"


def _is_logging_rank(rank: Optional[int] = None) -> bool:
    """True if this process should log to MLflow (rank 0 or single process)."""
    if rank is not None:
        return rank == 0
    env_rank = os.environ.get("RANK")
    return int(env_rank) == 0 if env_rank is not None else True


def _resolve_run_mode(run_mode: RunMode) -> RunMode:
    """Resolve run_mode: decorator value > context config > "auto"."""
    if run_mode != "auto":
        return run_mode
    config = get_mlflow_context()
    if config and config.run_mode != "auto":
        return config.run_mode
    return "auto"


def _get_flyte_tags() -> dict[str, str]:
    """Build MLflow tags from Flyte execution context."""
    ctx = flyte.ctx()
    if ctx is None or ctx.action is None:
        return {}

    tags = {}
    for attr, tag in [
        ("name", "flyte.action_name"),
        ("run_name", "flyte.run_name"),
        ("project", "flyte.project"),
        ("domain", "flyte.domain"),
    ]:
        val = getattr(ctx.action, attr, None)
        if val:
            tags[tag] = val
    return tags


def _setup_tracking(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
):
    """Configure MLflow tracking URI and experiment.

    Priority: explicit arg > mlflow_config() > env var.
    """
    config = get_mlflow_context()

    uri = tracking_uri or (config.tracking_uri if config else None)
    if uri:
        logger.info("Setting MLflow tracking URI: %s", uri)
        mlflow.set_tracking_uri(uri)

    exp_id = experiment_id or (config.experiment_id if config else None)
    exp_name = experiment_name or (config.experiment_name if config else None)

    if exp_id:
        logger.info("Setting MLflow experiment by ID: %s", exp_id)
        mlflow.set_experiment(experiment_id=exp_id)
    elif exp_name:
        logger.info("Setting MLflow experiment by name: %s", exp_name)
        mlflow.set_experiment(exp_name)


def _setup_autolog(
    autolog: bool = False,
    framework: Optional[str] = None,
    log_models: Optional[bool] = None,
    log_datasets: Optional[bool] = None,
    autolog_kwargs: Optional[dict[str, Any]] = None,
):
    """Enable MLflow autologging if requested.

    Enabled when decorator passes `autolog=True` or
    `mlflow_config(autolog=True)` is set in context.
    Decorator args take priority over context config.
    """
    config = get_mlflow_context()

    if not autolog and not (config and config.autolog):
        return

    if config:
        framework = framework if framework is not None else config.framework
        log_models = log_models if log_models is not None else config.log_models
        log_datasets = log_datasets if log_datasets is not None else config.log_datasets
        config_autolog_kwargs = config.autolog_kwargs or {}
    else:
        config_autolog_kwargs = {}

    # Merge: config autolog_kwargs (base) + decorator autolog_kwargs (override)
    merged = {
        "log_models": True if log_models is None else log_models,
        "log_datasets": True if log_datasets is None else log_datasets,
        **config_autolog_kwargs,
        **(autolog_kwargs or {}),
    }

    if framework:
        module = getattr(mlflow, framework, None)
        if module and hasattr(module, "autolog"):
            module.autolog(**merged)
        else:
            raise ValueError(f"MLflow framework '{framework}' not supported")
    else:
        mlflow.autolog(**merged)


def _start_run_kwargs_from_config() -> dict[str, Any]:
    """Build mlflow.start_run() kwargs from mlflow_config() context."""
    config = get_mlflow_context()
    if not config:
        return {}

    config_dict = asdict(config)
    extra = config_dict.pop("kwargs", None) or {}

    for key in _TRACKING_FIELDS | _PLUGIN_FIELDS:
        config_dict.pop(key, None)

    filtered = {k: v for k, v in config_dict.items() if v is not None}
    return {**extra, **filtered}


@contextmanager
def _run_for_plain_function(
    *,
    autolog: bool = False,
    framework: Optional[str] = None,
    log_models: Optional[bool] = None,
    log_datasets: Optional[bool] = None,
    autolog_kwargs: Optional[dict[str, Any]] = None,
    **start_kwargs,
):
    """MLflow run for plain functions.

    Creates a run, optionally enables autologging, and ends the run on exit.
    """
    logger.debug("Starting MLflow run for plain function (no Flyte context)")

    _setup_tracking(
        tracking_uri=start_kwargs.pop("tracking_uri", None),
        experiment_name=start_kwargs.pop("experiment_name", None),
        experiment_id=start_kwargs.pop("experiment_id", None),
    )

    run = mlflow.start_run(**start_kwargs)
    _setup_autolog(
        autolog=autolog,
        framework=framework,
        log_models=log_models,
        log_datasets=log_datasets,
        autolog_kwargs=autolog_kwargs,
    )

    try:
        yield run
    finally:
        mlflow.end_run()


@contextmanager
def _run_for_task(
    *,
    run_mode: RunMode = "auto",
    autolog: bool = False,
    framework: Optional[str] = None,
    log_models: Optional[bool] = None,
    log_datasets: Optional[bool] = None,
    autolog_kwargs: Optional[dict[str, Any]] = None,
    link_host: Optional[str] = None,
    link_template: Optional[str] = None,
    **start_kwargs,
):
    """MLflow run for Flyte tasks with full run_mode support.

    Handles tracking setup, run creation/reuse, autologging, and cleanup.
    """
    ctx = flyte.ctx()

    # Trace accessing parent's run — yield without creating a new run
    existing_run = ctx.data.get("_mlflow_run")
    if existing_run:
        logger.debug("Trace accessing parent's MLflow run: %s", existing_run.info.run_id)
        yield existing_run
        return

    # Save state to restore after this run
    saved_run_id = ctx.custom_context.get("_mlflow_run_id")
    saved_run = ctx.data.get("_mlflow_run")

    # Configure tracking (pops tracking fields from start_kwargs)
    _setup_tracking(
        tracking_uri=start_kwargs.pop("tracking_uri", None),
        experiment_name=start_kwargs.pop("experiment_name", None),
        experiment_id=start_kwargs.pop("experiment_id", None),
    )

    # Merge: context config (base) + decorator kwargs (override)
    merged = {**_start_run_kwargs_from_config(), **start_kwargs}

    current_action = ctx.action.name
    should_reuse = run_mode == "auto" and saved_run_id

    if should_reuse:
        reuse_run_id = merged.get("run_id") or saved_run_id

        # If the run is already active in this process (local execution),
        # just reuse it directly instead of calling start_run() again.
        active = mlflow.active_run()
        if active and active.info.run_id == reuse_run_id:
            logger.info(
                "Reusing already-active MLflow run: %s (run_mode=%s)",
                reuse_run_id,
                run_mode,
            )
            ctx.data["_mlflow_run"] = active

            _setup_autolog(
                autolog=autolog,
                framework=framework,
                log_models=log_models,
                log_datasets=log_datasets,
                autolog_kwargs=autolog_kwargs,
            )

            try:
                yield active
            finally:
                # Restore previous state — don't end the run, parent owns it
                if saved_run is not None:
                    ctx.data["_mlflow_run"] = saved_run
                else:
                    ctx.data.pop("_mlflow_run", None)
            return

        logger.info("Reusing MLflow run: %s (run_mode=%s)", reuse_run_id, run_mode)
        merged["run_id"] = reuse_run_id
    else:
        # Clear stale parent link for independent runs.
        # Nested runs keep the parent link — they're part of the same experiment.
        if run_mode != "nested":
            ctx.custom_context.pop("_mlflow_link", None)

        # Set run_name if not set already
        if not merged.get("run_name"):
            merged["run_name"] = f"{ctx.action.run_name}-{current_action}"

        # Set tags
        merged["tags"] = {**_get_flyte_tags(), **(merged.get("tags") or {})}

        # Nested mode: set mlflow.parentRunId tag so the MLflow UI shows
        # this run as a child of the parent. This works across processes
        # (no need to resume the parent run in this process).
        if run_mode == "nested":
            parent_run_id = saved_run_id
            if parent_run_id:
                merged["tags"]["mlflow.parentRunId"] = parent_run_id
                logger.info("Nesting under parent run: %s", parent_run_id)
            else:
                logger.warning("run_mode='nested' but no parent run ID found; creating a top-level run instead")

    # In local execution, tasks share a process. If a parent run is already
    # active and we're creating a new/nested run, MLflow requires nested=True.
    if not should_reuse and mlflow.active_run():
        merged["nested"] = True

    logger.info(
        "Starting MLflow run (run_mode=%s, action=%s, run_name=%s)",
        run_mode,
        current_action,
        merged.get("run_name", "<reusing>"),
    )

    run = mlflow.start_run(**merged)
    logger.info("MLflow run started: %s", run.info.run_id)

    ctx.custom_context["_mlflow_run_id"] = run.info.run_id
    ctx.data["_mlflow_run"] = run

    # Auto-set link if link_host is configured (decorator arg or config)
    config = get_mlflow_context()
    host = link_host or (config.link_host if config else None)
    if host:
        template = link_template or (config.link_template if config else None) or _DEFAULT_LINK_TEMPLATE
        ctx.custom_context["_mlflow_link"] = template.format(
            host=host.rstrip("/"),
            experiment_id=run.info.experiment_id,
            run_id=run.info.run_id,
        )

    _setup_autolog(
        autolog=autolog,
        framework=framework,
        log_models=log_models,
        log_datasets=log_datasets,
        autolog_kwargs=autolog_kwargs,
    )

    try:
        yield run
    finally:
        is_owner = run_mode in ("new", "nested") or (run_mode == "auto" and not saved_run_id)

        if run and (is_owner or not should_reuse):
            try:
                mlflow.end_run(status="FINISHED")
                logger.info("MLflow run ended: %s (FINISHED)", run.info.run_id)
            except Exception:
                try:
                    mlflow.end_run(status="FAILED")
                    logger.warning("MLflow run ended: %s (FAILED)", run.info.run_id)
                except Exception:
                    pass
                raise

        # Restore previous state
        if saved_run_id is not None:
            ctx.custom_context["_mlflow_run_id"] = saved_run_id
        else:
            ctx.custom_context.pop("_mlflow_run_id", None)

        if saved_run is not None:
            ctx.data["_mlflow_run"] = saved_run
        else:
            ctx.data.pop("_mlflow_run", None)


def _make_plain_func_wrapper(fn, *, run_mode, autolog, rank, run_kwargs):
    """Create a sync or async wrapper for plain functions."""

    def _open_run():
        ctx = flyte.ctx()
        if ctx:
            raise RuntimeError(
                "@mlflow_run cannot be applied to traces. "
                "Traces can access the parent's MLflow run via get_mlflow_run()."
            )

        # Copy so each invocation gets its own dict (context managers pop keys)
        return _run_for_plain_function(autolog=autolog, **dict(run_kwargs))

    if iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kw):
            if not _is_logging_rank(rank):
                return await fn(*args, **kw)
            with _open_run():
                return await fn(*args, **kw)

        return wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if not _is_logging_rank(rank):
            return fn(*args, **kw)
        with _open_run():
            return fn(*args, **kw)

    return wrapper


def mlflow_run(
    _func: Optional[F] = None,
    *,
    run_mode: RunMode = "auto",
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    autolog: bool = False,
    framework: Optional[str] = None,
    log_models: Optional[bool] = None,
    log_datasets: Optional[bool] = None,
    autolog_kwargs: Optional[dict[str, Any]] = None,
    rank: Optional[int] = None,
    **kwargs,
) -> F:
    """Decorator to manage MLflow runs for Flyte tasks and plain functions.

    Handles both manual logging and autologging. For autologging, pass
    `autolog=True` and optionally `framework` to select a specific
    framework (e.g. `"sklearn"`).

    Args:
        run_mode: "auto" (default), "new", or "nested".
            - "auto": reuse parent run if available, else create new.
            - "new": always create a new independent run.
            - "nested": create a new run nested under the parent via
              `mlflow.parentRunId` tag. Works across processes/containers.
        tracking_uri: MLflow tracking server URL.
        experiment_name: MLflow experiment name (exclusive with experiment_id).
        experiment_id: MLflow experiment ID (exclusive with experiment_name).
        run_name: Human-readable run name (exclusive with run_id).
        run_id: MLflow run ID (exclusive with run_name).
        tags: Dictionary of tags for the run.
        autolog: Enable MLflow autologging.
        framework: MLflow framework name for autolog (e.g. "sklearn", "pytorch").
        log_models: Whether to log models automatically (requires autolog).
        log_datasets: Whether to log datasets automatically (requires autolog).
        autolog_kwargs: Extra parameters passed to `mlflow.autolog()`.
        rank: Process rank for distributed training (only rank 0 logs).
        **kwargs: Additional `mlflow.start_run()` parameters.

    Decorator Order:
        @mlflow_run must be the outermost decorator::

            @mlflow_run
            @env.task
            async def my_task():
                ...
    """
    if experiment_name and experiment_id:
        raise ValueError("Cannot provide both 'experiment_name' and 'experiment_id'. Use one or the other.")
    if run_name and run_id:
        raise ValueError("Cannot provide both 'run_name' and 'run_id'. Use one or the other.")

    def decorator(func: F) -> F:
        run_kwargs: dict[str, Any] = {}
        for key, val in [
            ("tracking_uri", tracking_uri),
            ("experiment_name", experiment_name),
            ("experiment_id", experiment_id),
            ("run_name", run_name),
            ("run_id", run_id),
            ("tags", tags),
            ("framework", framework),
        ]:
            if val:
                run_kwargs[key] = val

        if log_models is not None:
            run_kwargs["log_models"] = log_models
        if log_datasets is not None:
            run_kwargs["log_datasets"] = log_datasets
        if autolog_kwargs:
            run_kwargs["autolog_kwargs"] = autolog_kwargs
        run_kwargs.update(kwargs)

        # Task template — wrap func.func (not func.execute) so that
        # mlflow.start_run() runs in the same thread as the task function.
        # Flyte runs sync tasks in a separate thread via run_sync_with_loop;
        # MLflow uses threading.local for its active run stack, so starting
        # the run in the async execute thread would be invisible to the task.
        if isinstance(func, AsyncFunctionTaskTemplate):
            # Set run_mode on Mlflow link instances at decoration time
            # so get_link() (called before execution) can suppress stale parent links
            for link in func.links:
                if isinstance(link, Mlflow):
                    if run_mode == "new":
                        link._decorator_run_mode = "new"
                    elif run_mode == "nested":
                        link._decorator_run_mode = "nested"
                        link.name = "MLflow (parent)"

            original_fn = func.func

            if iscoroutinefunction(original_fn):

                @functools.wraps(original_fn)
                async def wrapped_fn(*args, **kw):
                    if not _is_logging_rank(rank):
                        return await original_fn(*args, **kw)
                    with _run_for_task(
                        run_mode=_resolve_run_mode(run_mode),
                        autolog=autolog,
                        **dict(run_kwargs),
                    ):
                        return await original_fn(*args, **kw)

            else:

                @functools.wraps(original_fn)
                def wrapped_fn(*args, **kw):
                    if not _is_logging_rank(rank):
                        return original_fn(*args, **kw)
                    with _run_for_task(
                        run_mode=_resolve_run_mode(run_mode),
                        autolog=autolog,
                        **dict(run_kwargs),
                    ):
                        return original_fn(*args, **kw)

            func.func = wrapped_fn
            return cast(F, func)

        # Plain function (sync or async) — HPO objectives, traces, etc.
        return cast(
            F,
            _make_plain_func_wrapper(
                func,
                run_mode=run_mode,
                autolog=autolog,
                rank=rank,
                run_kwargs=run_kwargs,
            ),
        )

    if _func is None:
        return decorator

    return decorator(_func)
