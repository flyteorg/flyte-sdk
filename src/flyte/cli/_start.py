import rich_click as click

_REMOTE_TUI_INSTALL_HINT = (
    "The remote TUI requires flyteplugins-remote-tui in the same Python environment as flyte.\n\n"
    "  pip install flyteplugins-remote-tui\n\n"
    "If flyte was installed with uv tool install, add the plugin to that environment:\n"
    "  uv tool install flyte --with flyteplugins-remote-tui --force"
)


@click.group()
def start():
    """Start various Flyte services."""


@start.command()
def tui():
    """
    Launch TUI explore mode to browse past local runs. To use the TUI install `pip install flyte[tui]`
    TUI, allows you to explore all your local runs if you have persistence enabled.

    Persistence can be enabled in 2 ways,
    1. By setting it in the config to record every local run
    ```bash
    flyte create config --endpoint ...  --local-persistence
    ```
    2. By passing it in flyte.init(local_persistence=True)
    This will record all `flyte.run` runs, that are local and are within the flyte.init being active.
    """
    from flyte.cli._tui import launch_tui_explore

    launch_tui_explore()


@start.command(name="remote-tui")
@click.option(
    "-c",
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the Flyte configuration file. Defaults to ~/.flyte/config.yaml.",
)
@click.option(
    "--poll-interval",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds between run detail refreshes while a run is active.",
)
def remote_tui(config_file: str | None, poll_interval: float) -> None:
    """
    Interactive TUI for a remote Flyte v2 cluster.

    Browse runs, actions, logs, tasks, apps, and triggers. Requires a Flyte config file
    (``flyte create config``) and ``flyteplugins-remote-tui``.
    """
    try:
        from flyteplugins.remote_tui import launch_remote_tui
    except ImportError as exc:
        raise click.ClickException(_REMOTE_TUI_INSTALL_HINT) from exc

    launch_remote_tui(config=config_file, poll_interval=poll_interval)


_DEFAULT_DEVBOX_IMAGE = "cr.flyte.org/flyteorg/flyte-devbox:latest"
_DEFAULT_DEVBOX_GPU_IMAGE = "cr.flyte.org/flyteorg/flyte-devbox:gpu-latest"


@start.command()
@click.option(
    "--image",
    default=None,
    show_default=f"{_DEFAULT_DEVBOX_IMAGE} ({_DEFAULT_DEVBOX_GPU_IMAGE} when --gpu)",
    help="Docker image to use for the devbox cluster.",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Enable dev mode inside the devbox cluster (sets FLYTE_DEV=True).",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=False,
    help="Pass host GPUs into the devbox container (adds --gpus all to docker run). "
    "Requires an NVIDIA-enabled host. Defaults --image to a GPU-capable image "
    "if --image is not explicitly set.",
)
@click.pass_context
def devbox(ctx: click.Context, image: str | None, dev: bool, gpu: bool):
    """Start a local Flyte devbox cluster."""
    from flyte._sentry import count
    from flyte.cli._devbox import launch_devbox

    if image is None:
        image = _DEFAULT_DEVBOX_GPU_IMAGE if gpu else _DEFAULT_DEVBOX_IMAGE

    count("cli.command", command="start_devbox")
    log_format = getattr(ctx.obj, "log_format", "console") if ctx.obj else "console"
    launch_devbox(image, dev, gpu=gpu, log_format=log_format)
