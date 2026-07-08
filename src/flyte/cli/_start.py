import rich_click as click


@click.group()
def start():
    """Start various Flyte services."""


@start.command()
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
    help="Seconds between run detail refreshes while browsing a remote run. Remote mode only.",
)
def tui(config_file: str | None, poll_interval: float) -> None:
    """
    Launch the Flyte TUI. Install with ``pip install flyte[tui]``.

    The mode is chosen from the resolved config:

    * Remote (config has an endpoint, or FLYTE_API_KEY is set): browse a remote
      Flyte v2 cluster — projects, runs, actions, logs, tasks, apps, and triggers.
      ``flyte start tui --config remote.yaml``
    * Local (no endpoint): explore past local runs recorded with persistence.
      ``flyte start tui --config local.yaml``

    Local persistence can be enabled in 2 ways:

    1. In the config, to record every local run:
       ``flyte create config --endpoint ... --local-persistence``
    2. Via ``flyte.init(local_persistence=True)``, recording ``flyte.run`` runs
       that are local and within the active ``flyte.init``.
    """
    from flyte.cli._tui import config_is_remote, launch_tui_explore, launch_tui_remote

    if config_is_remote(config_file):
        launch_tui_remote(config=config_file, poll_interval=poll_interval)
    else:
        launch_tui_explore()


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
