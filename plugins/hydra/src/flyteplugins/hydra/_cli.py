"""``flyte hydra`` CLI command group.

Registered via the ``flyte.plugins.cli.commands`` entry point so that
``flyte hydra run`` is available once ``flyteplugins-hydra`` is installed.

Inherits the standard ``flyte run`` flags that apply to script execution
(``--project``, ``--domain``, ``--local``, ``--image``, ``--follow``, etc.).
Hydra-specific options are:
``--config-path``, ``--config-name``, ``--mode``, ``--multirun``,
``--wait/--no-wait``, ``--wait-max-workers``, ``--task-env-key``,
``--hydra-override``.
Application config overrides use the task's ``DictConfig`` parameter name,
for example ``--cfg`` for ``cfg: DictConfig`` or ``--config`` for
``config: DictConfig``.

Usage
-----
Single run (remote by default)::

    flyte hydra run --config-path conf --config-name training \\
        train.py pipeline \\
        --cfg optimizer.lr=0.01

Single run forced local::

    flyte hydra run --local --config-path conf --config-name training \\
        train.py pipeline

Grid sweep (six parallel remote executions)::

    flyte hydra run --multirun --config-path conf --config-name training \\
        train.py pipeline \\
        --cfg "optimizer.lr=0.001,0.01,0.1" --cfg "training.epochs=10,20"

TPE/Bayesian sweep via Optuna sweeper::

    flyte hydra run --multirun --config-path conf --config-name training \\
        train.py pipeline \\
        --hydra-override hydra/sweeper=optuna \\
        --hydra-override hydra.sweeper.n_trials=20 \\
        --hydra-override hydra.sweeper.n_jobs=4 \\
        --cfg "optimizer.lr=interval(1e-4,1e-1)"
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import rich_click as click


def _follow_run_logs(run) -> None:
    """Show logs for a returned remote Run when ``--follow`` is set.

    ``hydra_run`` / ``hydra_sweep`` return whatever ``flyte.run`` returned.
    In remote mode that should be a ``flyte.remote.Run`` with ``show_logs``;
    in local mode it is a local result wrapper, so this helper quietly skips
    objects that do not expose remote logs.
    """
    show_logs = getattr(run, "show_logs", None)
    if show_logs is not None:
        show_logs(max_lines=30, show_ts=True, raw=False)


def _completed_result_value(run):
    """Return a completed result value without re-printing remote run URLs."""
    if hasattr(run, "value"):
        return run.value
    if getattr(run, "url", None) is None:
        return run
    return None


def _extract_config_overrides(task, args: list[str]) -> tuple[list[str], list[str]]:
    """Split DictConfig override flags out of the task-argument tail.

    ``flyte hydra run`` names application config override flags after the
    task's ``DictConfig`` input. For ``cfg: DictConfig`` users pass
    ``--cfg optimizer.lr=0.01``; for ``config: DictConfig`` they pass
    ``--config optimizer.lr=0.01``. These flags sit after ``SCRIPT TASK_NAME``
    beside ordinary task args, so Click cannot parse them with fixed command
    options. This helper scans that tail, returns the extracted Hydra override
    strings, and leaves all other args for normal Flyte task-parameter parsing.
    """
    from flyteplugins.hydra._run import _config_param_names

    config_param_names = set(_config_param_names(task))
    if not config_param_names:
        return [], args

    config_options = {f"--{name}" for name in config_param_names}
    overrides: list[str] = []
    remaining: list[str] = []
    idx = 0

    while idx < len(args):
        arg = args[idx]
        matched = False

        for option in config_options:
            # Support both "--config-param value" and "--config-param=value".
            if arg == option:
                if idx + 1 >= len(args):
                    raise click.UsageError(
                        f"Option '{option}' requires an override value."
                    )
                overrides.append(args[idx + 1])
                idx += 2
                matched = True
                break
            if arg.startswith(f"{option}="):
                overrides.append(arg.split("=", 1)[1])
                idx += 1
                matched = True
                break

        if matched:
            continue

        remaining.append(arg)
        idx += 1

    return overrides, remaining


def _parse_task_kwargs(task, args: list[str], parent_ctx: click.Context) -> dict:
    """Convert ordinary task CLI flags into Python kwargs for ``flyte.run``.

    The Hydra command has to load the user script before it can know the task
    interface. Once the task is available, this function builds a temporary
    Click command from Flyte's typed interface using the same ``to_click_option``
    converters as ``flyte run``. ``DictConfig`` inputs are intentionally
    skipped because those are composed by Hydra and injected by ``hydra_run`` /
    ``hydra_sweep``.
    """
    from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
    from flyte.cli._params import to_click_option
    from flyteplugins.hydra._run import _config_param_names

    interface = transform_native_to_typed_interface(task.native_interface)
    if interface is None:
        return {}

    inputs_interface = task.native_interface.inputs
    config_param_names = set(_config_param_names(task))
    params = []

    for entry in interface.inputs.variables:
        name, var = entry.key, entry.value
        if name in config_param_names:
            continue

        default_val = None
        if inputs_interface[name][1] is not inspect._empty:
            default_val = inputs_interface[name][1]

        params.append(
            to_click_option(name, var, inputs_interface[name][0], default_val)
        )

    def _collect(**kwargs):
        return kwargs

    # Let Click apply Flyte's normal type conversion and required/default
    # validation for the remaining task inputs.
    parser = click.Command(
        name="task-args",
        params=params,
        callback=_collect,
    )
    return parser.main(
        args=args,
        prog_name="task arguments",
        standalone_mode=False,
        obj=parent_ctx.obj,
    )


@click.group(name="hydra")
def hydra_group() -> None:
    """Run Flyte tasks via Hydra config composition and sweeping."""


@hydra_group.command(
    name="run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.argument("task_name")
@click.option("--config-path", default=None, help="Path to Hydra config directory.")
@click.option(
    "--config-name",
    default="config",
    show_default=True,
    help="Top-level config file name (without .yaml).",
)
@click.option(
    "--mode",
    type=click.Choice(["local", "remote"]),
    default=None,
    help="Execution mode. Defaults to remote for Flyte CLI parity.",
)
@click.option(
    "--multirun",
    is_flag=True,
    default=False,
    help="Expand sweep overrides into a grid of executions.",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    show_default=True,
    help="Wait for remote Flyte runs to reach a terminal phase.",
)
@click.option(
    "--wait-max-workers",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Maximum worker threads used while waiting for remote Flyte runs.",
)
@click.option(
    "--task-env-key",
    default="task_env",
    show_default=True,
    help="Config key containing entry-task task.override kwargs by task name.",
)
@click.option(
    "--hydra-override",
    "hydra_overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Hydra-namespace override (repeatable), e.g. hydra/sweeper=optuna or "
        "hydra.sweeper.n_trials=20."
    ),
)
@click.pass_context
def hydra_run_cmd(
    ctx: click.Context,
    script: str,
    task_name: str,
    config_path: str | None,
    config_name: str,
    mode: str | None,
    multirun: bool,
    wait: bool,
    wait_max_workers: int,
    task_env_key: str,
    hydra_overrides: tuple[str, ...],
    **run_params,
) -> None:
    """Compose a Hydra config and run TASK_NAME from SCRIPT on Flyte.

    SCRIPT is the path to a Python file containing the Flyte task.
    TASK_NAME is the name of the task function to run.

    Use the task's ``DictConfig`` parameter name for app-level overrides
    (for example ``--cfg`` or ``--config``).
    Use ``--hydra-override`` for hydra-namespace settings (hydra/sweeper=optuna).
    """
    from flyte.cli._common import initialize_config
    from flyte.cli._run import RunArguments

    run_args = RunArguments.from_dict(run_params)

    # Initialise Flyte.
    config = initialize_config(
        ctx,
        run_args.project,
        run_args.domain,
        root_dir=run_args.root_dir,
        images=run_args.image or None,
        sync_local_sys_paths=not run_args.no_sync_local_sys_paths,
    )
    ctx.obj = config.replace(run_args=run_args)

    if run_args.local and mode == "remote":
        raise click.UsageError("Use either --local or --mode remote, not both.")
    if run_args.follow and not wait:
        raise click.UsageError("Use either --follow or --no-wait, not both.")

    execution_mode = "local" if run_args.local else (mode or "remote")

    # Only forward options that are accepted by flyte.with_runcontext and are
    # meaningful for an in-process script task. Project/domain/image/root-dir
    # are handled by initialize_config above; follow is handled after launch.
    run_options: dict = {
        "log_format": config.log_format,
        "reset_root_logger": config.reset_root_logger,
    }
    if run_args.service_account is not None:
        run_options["service_account"] = run_args.service_account
    if run_args.name is not None:
        run_options["name"] = run_args.name
    if run_args.raw_data_path is not None:
        run_options["raw_data_path"] = run_args.raw_data_path
    if run_args.copy_style != "loaded_modules":
        run_options["copy_style"] = run_args.copy_style
    if run_args.debug:
        run_options["debug"] = True

    # Load the script as a module so Flyte task decorators have run before we
    # inspect the requested task's typed interface.
    script_path = Path(script).resolve()
    module_name = script_path.stem
    sys.path.append(str(script_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load module from {script}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    task = getattr(mod, task_name, None)
    if task is None:
        raise click.ClickException(f"Task '{task_name}' not found in {script}")

    # ctx.args contains everything after SCRIPT TASK_NAME that was not consumed
    # by the fixed Hydra/Flyte options. First pull out DictConfig override
    # aliases, then parse the rest as ordinary task inputs.
    param_cfg_overrides, task_args = _extract_config_overrides(task, list(ctx.args))
    task_kwargs = _parse_task_kwargs(task, task_args, ctx)

    # Combine all overrides — hydra_sweep / hydra_run handle separation
    # of hydra-namespace vs app-level overrides internally.
    all_overrides = param_cfg_overrides + list(hydra_overrides)

    if multirun:
        from flyteplugins.hydra._run import hydra_sweep

        runs = hydra_sweep(
            task,
            config_path=config_path,
            config_name=config_name,
            overrides=all_overrides,
            mode=execution_mode,
            wait=wait,
            wait_max_workers=wait_max_workers,
            run_options=run_options or None,
            task_env_key=task_env_key,
            **task_kwargs,
        )
        for i, run in enumerate(runs):
            value = _completed_result_value(run)
            if value is not None:
                click.echo(f"[{i}] result={value}")
            if run_args.follow and execution_mode == "remote":
                _follow_run_logs(run)
    else:
        from flyteplugins.hydra._run import hydra_run

        run = hydra_run(
            task,
            config_path=config_path,
            config_name=config_name,
            overrides=all_overrides,
            mode=execution_mode,
            wait=wait,
            wait_max_workers=wait_max_workers,
            run_options=run_options or None,
            task_env_key=task_env_key,
            **task_kwargs,
        )
        value = _completed_result_value(run)
        if value is not None:
            click.echo(value)
        if run_args.follow and execution_mode == "remote":
            _follow_run_logs(run)


# Dynamically inherit all standard ``flyte run`` options.
# Reuses RunArguments.options() so that new options added to ``flyte run``
# are automatically available on ``flyte hydra run`` without duplication.
# If a future ``flyte run`` option collides with a hydra-specific flag, the
# import fails immediately with a clear error rather than silently breaking.
def _extend_with_run_options() -> None:
    from flyte.cli._run import RunArguments

    hydra_option_names = {p.name for p in hydra_run_cmd.params}
    unsupported_options = {
        "run_project",
        "run_domain",
        "tui",
    }
    for opt in RunArguments.options():
        if opt.name in unsupported_options:
            continue
        if opt.name in hydra_option_names:
            raise RuntimeError(
                f"flyte run option '{opt.name}' conflicts with a hydra-specific "
                f"option on 'flyte hydra run'. The flyteplugins-hydra plugin "
                f"needs to be updated to resolve this collision."
            )
        hydra_run_cmd.params.append(opt)


_extend_with_run_options()
