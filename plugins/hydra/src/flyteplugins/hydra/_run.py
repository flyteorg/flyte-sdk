"""hydra_run / hydra_sweep — Python SDK entry points for the Flyte Hydra plugin.

These are convenience wrappers for users who prefer calling from Python rather
than writing a ``@hydra.main`` script. Both delegate to ``FlyteLauncher`` for
mode management, so single runs and sweeps share exactly the same code path.

Examples::

    from flyteplugins.hydra import hydra_run, hydra_sweep

    # Single run, config from YAML
    hydra_run(pipeline, config_path="conf", config_name="training",
              dataset="s3://bucket/imagenet", mode="remote")

    # Grid sweep — one Flyte execution per combination
    runs = hydra_sweep(
        pipeline,
        config_path="conf", config_name="training",
        overrides=["optimizer.lr=0.001,0.01,0.1", "training.epochs=10,20"],
        dataset="s3://bucket/imagenet",
        mode="remote",
    )

    # TPE/Bayesian sweep via Optuna sweeper
    runs = hydra_sweep(
        pipeline,
        config_path="conf", config_name="training",
        overrides=[
            "hydra/sweeper=optuna", "hydra.sweeper.n_trials=20",
            "hydra.sweeper.n_jobs=4",
            "optimizer.lr=interval(1e-4,1e-1)",
        ],
        mode="remote",
    )
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import flyte
from omegaconf import DictConfig, OmegaConf

from hydra import initialize, initialize_config_dir

_PRIMARY_CONTAINER_NAME_KEY = "primary_container_name"


def _is_dictconfig_type(tp: Any) -> bool:
    """Return True when *tp* is OmegaConf's DictConfig type."""
    try:
        return isinstance(tp, type) and issubclass(tp, DictConfig)
    except TypeError:
        return False


def _task_inputs(task: Callable) -> dict[str, tuple[Any, Any]]:
    """Return task inputs from Flyte's native interface when available."""
    native_interface = getattr(task, "native_interface", None)
    if native_interface is not None:
        return native_interface.inputs

    sig = inspect.signature(task)
    return {name: (param.annotation, param.default) for name, param in sig.parameters.items()}


def _config_param_names(task: Callable) -> list[str]:
    """Return DictConfig input parameter names declared on *task*."""
    return [name for name, (param_type, _) in _task_inputs(task).items() if _is_dictconfig_type(param_type)]


def _config_param_name(task: Callable) -> str:
    """Return the DictConfig parameter to inject, defaulting to ``cfg``."""
    names = _config_param_names(task)
    if names:
        return names[0]
    return "cfg"


def _task_name(task: Callable) -> str:
    """Return the user-facing task name used for task-env config lookups."""
    return getattr(task, "short_name", None) or getattr(task, "__name__", None) or getattr(task, "name", None) or "task"


def _pod_template_with_image(
    image: str,
    *,
    primary_container_name: str,
    pod_template: flyte.PodTemplate | None = None,
) -> flyte.PodTemplate:
    """Return a PodTemplate whose primary container uses a prebuilt image URI."""
    from kubernetes.client import V1Container, V1PodSpec

    if not isinstance(image, str):
        raise TypeError(f"Expected task_env image to be a prebuilt image URI string. Got {type(image).__name__}.")

    if pod_template is None:
        return flyte.PodTemplate(
            pod_spec=V1PodSpec(containers=[V1Container(name=primary_container_name, image=image)]),
            primary_container_name=primary_container_name,
        )

    merged = copy.deepcopy(pod_template)
    merged.primary_container_name = primary_container_name
    if merged.pod_spec is None:
        merged.pod_spec = V1PodSpec(containers=[])
    if merged.pod_spec.containers is None:
        merged.pod_spec.containers = []

    for container in merged.pod_spec.containers:
        if container.name == primary_container_name:
            container.image = image
            break
    else:
        merged.pod_spec.containers.append(V1Container(name=primary_container_name, image=image))

    return merged


def _coerce_override_kwargs(overrides: Any) -> dict[str, Any]:
    """Normalize Hydra task-env mappings before applying them to a task.

    Most keys are passed through to ``task.override``. ``resources`` mappings
    are converted to ``flyte.Resources``. ``image`` and
    ``primary_container_name`` are plugin-level conveniences that are resolved
    later, once the launched task's existing pod template is available.
    """
    if isinstance(overrides, DictConfig):
        kwargs = OmegaConf.to_container(overrides, resolve=True)
    elif isinstance(overrides, Mapping):
        kwargs = dict(overrides)
    else:
        kwargs = overrides

    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected task override kwargs to be a mapping, got {type(kwargs).__name__}.")

    # YAML/structured configs naturally express resources as a mapping, while
    # task.override expects a flyte.Resources object.
    resources = kwargs.get("resources")
    if isinstance(resources, Mapping):
        kwargs["resources"] = flyte.Resources(**dict(resources))

    return kwargs


def _merge_task_env_image(task: Callable, override_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve task-env ``image`` into a pod template override for task.

    ``task.override`` intentionally rejects ``image`` because the task image is
    part of the task definition. For Hydra task-env presets, we support
    prebuilt runtime images by setting the image on the pod template primary
    container. If the task already has an inline pod template, keep it and only
    patch a deep copy of its primary container.
    """
    if "image" not in override_kwargs:
        if _PRIMARY_CONTAINER_NAME_KEY in override_kwargs:
            raise ValueError(f"'{_PRIMARY_CONTAINER_NAME_KEY}' requires 'image'.")
        return override_kwargs

    kwargs = dict(override_kwargs)
    image = kwargs.pop("image")
    requested_primary_name = kwargs.pop(_PRIMARY_CONTAINER_NAME_KEY, None)

    # Users can still pass a Python-created pod template through the SDK path.
    # YAML examples avoid modeling full V1PodSpec values.
    configured_pod_template = kwargs.get("pod_template")
    base_pod_template = (
        configured_pod_template if configured_pod_template is not None else getattr(task, "pod_template", None)
    )

    if isinstance(base_pod_template, str):
        raise ValueError(
            "Cannot apply task_env image to a task with a named pod_template string. "
            "Use an inline flyte.PodTemplate or set the image in the referenced pod template."
        )

    if base_pod_template is not None and not isinstance(base_pod_template, flyte.PodTemplate):
        raise TypeError(
            "Expected task pod_template to be a flyte.PodTemplate when using task_env image, "
            f"got {type(base_pod_template).__name__}."
        )

    primary_container_name = requested_primary_name
    if primary_container_name is None and base_pod_template is not None:
        primary_container_name = base_pod_template.primary_container_name

    kwargs["pod_template"] = _pod_template_with_image(
        image,
        primary_container_name=primary_container_name or "primary",
        pod_template=base_pod_template,
    )
    return kwargs


def _task_override_kwargs(cfg: DictConfig, task_env_key: str, task_name: str) -> dict:
    """Return entry-task override kwargs from ``cfg[task_env_key][task_name]``.

    The launcher only controls the task it passes to ``flyte.run``. Child task
    overrides must be applied inside user code, where those child tasks are
    called. The returned mapping may still include plugin-level conveniences
    such as ``image``; ``_make_entry`` resolves those after it can inspect the
    launched task's existing pod template.
    """
    task_env = cfg.get(task_env_key, {})
    if not task_env:
        return {}

    if not isinstance(task_env, (DictConfig, Mapping)):
        raise TypeError(
            f"Expected '{task_env_key}' to be a mapping from task name to "
            f"task.override kwargs, got {type(task_env).__name__}."
        )

    task_overrides = task_env.get(task_name, {})
    if not task_overrides:
        return {}

    if not isinstance(task_overrides, (DictConfig, Mapping)):
        raise TypeError(
            f"Expected '{task_env_key}.{task_name}' to contain task.override "
            f"kwargs, got {type(task_overrides).__name__}."
        )
    return _coerce_override_kwargs(task_overrides)


def apply_task_env(
    task: Callable,
    cfg: DictConfig | Mapping[str, Any],
    *,
    task_env_key: str = "task_env",
    task_name: str | None = None,
) -> Callable:
    """Return task with Hydra task-env overrides applied.

    The launcher calls this for the entry task automatically. User code can call
    it for child tasks to get the same resources and prebuilt-image handling
    before invoking the task.
    """
    override_kwargs = _merge_task_env_image(
        task,
        _task_override_kwargs(cfg, task_env_key, task_name or _task_name(task)),
    )
    return task.override(**override_kwargs) if override_kwargs else task


def _task_kwargs(task: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter ``kwargs`` to declared task parameters, excluding DictConfig inputs."""
    inputs = _task_inputs(task)
    config_params = set(_config_param_names(task)) or {"cfg"}
    return {k: v for k, v in kwargs.items() if k in inputs and k not in config_params}


@contextlib.contextmanager
def _hydra_init(config_path: str | Path | None):
    """Initialize Hydra and yield the config loader.

    The caller is responsible for using the config loader within the ``with``
    block — GlobalHydra is cleared on exit.
    """
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    if config_path is not None:
        ctx = initialize_config_dir(config_dir=str(Path(config_path).absolute()), version_base=None)
    else:
        ctx = initialize(config_path=None, version_base=None)

    with ctx:
        yield GlobalHydra.instance().config_loader()


def _setup_launcher(
    config_loader,
    config_name,
    mode,
    wait,
    wait_max_workers,
    task_function,
    run_mode,
):
    """Create a ``FlyteLauncher``, load master config, and call ``setup``."""
    from flyteplugins.hydra._launcher import FlyteLauncher
    from hydra._internal.callbacks import Callbacks
    from hydra.types import HydraContext

    # Load the master config *without* user overrides — per-job overrides are
    # applied by ``load_sweep_config`` inside ``FlyteLauncher._sweep_config``.
    master_cfg = config_loader.load_configuration(
        config_name=config_name,
        overrides=[],
        run_mode=run_mode,
        from_shell=False,
    )

    hydra_ctx = HydraContext(
        config_loader=config_loader,
        callbacks=Callbacks(),
    )

    launcher = FlyteLauncher(
        mode=mode,
        wait=wait,
        wait_max_workers=wait_max_workers,
    )
    launcher.setup(
        hydra_context=hydra_ctx,
        task_function=task_function,
        config=master_cfg,
    )
    return launcher


def _make_entry(
    task: Callable,
    task_kw: dict[str, Any],
    run_options: dict[str, Any] | None,
    mode: str,
    task_env_key: str,
):
    """Build the task-function wrapper passed to FlyteLauncher / Hydra runtime.

    Applies task-environment overrides to the entry task being launched and
    delegates to ``flyte.with_runcontext().run``. The explicit "mode" keeps
    SDK/CLI behavior deterministic.
    """

    config_param = _config_param_name(task)
    task_name = _task_name(task)

    def _entry(cfg: DictConfig) -> Any:
        t = apply_task_env(task, cfg, task_env_key=task_env_key, task_name=task_name)
        run_kwargs = {config_param: cfg, **task_kw}
        return flyte.with_runcontext(mode=mode, **(run_options or {})).run(t, **run_kwargs)

    return _entry


def _needs_hydra_runtime(overrides: list[str]) -> bool:
    """Return True if *overrides* contain Hydra config group selections.

    Config group selections (``hydra/sweeper=optuna``,
    ``hydra/callbacks=custom``, etc.) require the full Hydra runtime for plugin
    discovery. Simple value
    overrides (``hydra.run.dir=...``) work fine without the runtime.
    """
    return any(o.startswith("hydra/") for o in overrides)


def _split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Separate hydra-namespace overrides from app-level overrides."""
    hydra_ovrs: list[str] = []
    app_ovrs: list[str] = []

    for o in overrides:
        if o.startswith(("hydra/", "hydra.")):
            hydra_ovrs.append(o)
        else:
            app_ovrs.append(o)

    return app_ovrs, hydra_ovrs


def _expand_basic_sweep_overrides(config_loader, overrides: list[str]) -> list[list[str]]:
    """Expand comma-separated BasicSweeper overrides into per-job strings."""
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create(config_loader=config_loader)
    parsed_overrides = parser.parse_overrides(overrides)
    batches = BasicSweeper.split_arguments(parsed_overrides, max_batch_size=None)

    if not batches:
        return [[]]
    return batches[0]


def _sweep_via_hydra_runtime(
    task: Callable,
    config_path: str | Path | None,
    config_name: str,
    app_overrides: list[str],
    hydra_overrides: list[str],
    mode: str,
    task_kw: dict[str, Any],
    run_options: dict[str, Any] | None,
    task_env_key: str,
    wait: bool,
    wait_max_workers: int | None,
) -> list[Any]:
    """Invoke the full Hydra runtime so custom sweepers are discovered.

    Calls Hydra's runtime entry point so ConfigStore-discovered plugins can be
    instantiated. The selected sweeper expands overrides into per-job sets
    before calling ``FlyteLauncher.launch()``.
    """
    from hydra._internal.utils import _run_hydra, get_args_parser

    collected: list[Any] = []
    _entry = _make_entry(task, task_kw, run_options, mode, task_env_key)

    def _collecting_entry(cfg: DictConfig) -> Any:
        result = _entry(cfg)
        collected.append(result)
        return result

    launcher_overrides = [
        "hydra/launcher=flyte",
        f"hydra.launcher.mode={mode}",
        f"hydra.launcher.wait={str(wait).lower()}",
        f"hydra.launcher.wait_max_workers={'null' if wait_max_workers is None else wait_max_workers}",
    ]
    overrides_argv = launcher_overrides + hydra_overrides + app_overrides

    args_parser = get_args_parser()
    args = args_parser.parse_args(["--multirun", *overrides_argv])

    _run_hydra(
        args=args,
        args_parser=args_parser,
        task_function=_collecting_entry,
        config_path=str(Path(config_path).absolute()) if config_path else None,
        config_name=config_name,
    )

    return collected


def hydra_run(
    task: Callable,
    *,
    config_path: str | Path | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    mode: str = "remote",
    wait: bool = True,
    wait_max_workers: int | None = 32,
    run_options: dict[str, Any] | None = None,
    task_env_key: str = "task_env",
    **kwargs: Any,
) -> Any:
    """Run a single Flyte task with a Hydra-composed config.

    Args:
        task: Flyte task that accepts a ``cfg: DictConfig`` parameter.
        config_path: Path to the config directory. ``None`` for structured-config-only use.
        config_name: Top-level config file name (without ``.yaml``).
        overrides: Hydra override strings, e.g. ``["optimizer.lr=0.01"]``.
        mode: ``"remote"`` (default) or ``"local"``.
        wait: Whether to wait for remote Flyte runs to reach a terminal phase.
        wait_max_workers: Max worker threads used to wait for remote runs.
        run_options: Optional dict of kwargs forwarded to ``flyte.with_runcontext``
            (e.g. ``service_account``, ``name``, ``copy_style``, ``raw_data_path``).
        task_env_key: Config key containing entry-task ``task.override`` kwargs,
            nested under the launched task's name. Child task overrides must be
            applied by user code.
        **kwargs: Additional task arguments (non-``cfg`` parameters).

    Returns:
        The task result. Waited remote runs return a wrapper with ``url`` and
        the resolved task output.
    """
    from hydra.types import RunMode

    overrides = overrides or []
    kw = _task_kwargs(task, kwargs)
    _entry = _make_entry(task, kw, run_options, mode, task_env_key)

    with _hydra_init(config_path) as config_loader:
        launcher = _setup_launcher(
            config_loader,
            config_name,
            mode,
            wait,
            wait_max_workers,
            _entry,
            RunMode.RUN,
        )
        results = launcher.launch([overrides], initial_job_idx=0)

    if results:
        return results[0]._return_value
    return None


def hydra_sweep(
    task: Callable,
    *,
    config_path: str | Path | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
    mode: str = "remote",
    wait: bool = True,
    wait_max_workers: int | None = 32,
    run_options: dict[str, Any] | None = None,
    task_env_key: str = "task_env",
    **kwargs: Any,
) -> list[Any]:
    """Run a Hydra sweep, one Flyte execution per override combination.

    Comma-separated values in ``overrides`` are expanded into a Cartesian
    product. For example::

        overrides=["optimizer.lr=0.001,0.01,0.1", "training.epochs=10,20"]

    produces six executions.

    Custom sweepers (e.g. Optuna) are supported — include sweeper overrides
    directly in the ``overrides`` list::

        overrides=[
            "hydra/sweeper=optuna", "hydra.sweeper.n_trials=20",
            "hydra.sweeper.n_jobs=4",
            "optimizer.lr=interval(1e-4,1e-1)",
        ]

    When a custom sweeper is detected, the full Hydra runtime is used so the
    sweeper plugin is properly discovered and invoked.

    Args:
        task: Flyte task that accepts a ``cfg: DictConfig`` parameter.
        config_path: Path to the config directory.
        config_name: Top-level config file name (without ``.yaml``).
        overrides: Hydra sweep override strings (app-level and/or hydra-namespace).
        mode: ``"remote"`` (default) or ``"local"``.
        wait: Whether to wait for remote Flyte runs to reach a terminal phase.
        wait_max_workers: Max worker threads used to wait for remote runs.
        run_options: Optional dict of kwargs forwarded to ``flyte.with_runcontext``
            (e.g. ``service_account``, ``name``, ``copy_style``, ``raw_data_path``).
        task_env_key: Config key containing entry-task ``task.override`` kwargs,
            nested under the launched task's name. Child task overrides must be
            applied by user code.
        **kwargs: Additional task arguments (non-``cfg`` parameters).

    Returns:
        List of job results. Waited remote runs return wrappers with ``url``
        and the resolved task outputs.
    """
    overrides = overrides or []
    kw = _task_kwargs(task, kwargs)
    app_overrides, hydra_overrides = _split_overrides(overrides)

    # Hydra config group selections (hydra/sweeper=..., hydra/callbacks=...,
    # etc.) need the full Hydra runtime for plugin discovery. Basic grid
    # sweeps go through FlyteLauncher directly.
    if _needs_hydra_runtime(hydra_overrides):
        return _sweep_via_hydra_runtime(
            task,
            config_path,
            config_name,
            app_overrides,
            hydra_overrides,
            mode,
            kw,
            run_options,
            task_env_key,
            wait,
            wait_max_workers,
        )

    from hydra.types import RunMode

    _entry = _make_entry(task, kw, run_options, mode, task_env_key)

    # For basic sweeps, hydra-namespace overrides (e.g. hydra.run.dir=...) are
    # still valid — they're applied per-job via load_sweep_config.
    all_overrides = app_overrides + hydra_overrides

    with _hydra_init(config_path) as config_loader:
        job_overrides = _expand_basic_sweep_overrides(config_loader, all_overrides)
        launcher = _setup_launcher(
            config_loader,
            config_name,
            mode,
            wait,
            wait_max_workers,
            _entry,
            RunMode.MULTIRUN,
        )
        results = launcher.launch(job_overrides, initial_job_idx=0)

    return [r._return_value for r in results]
