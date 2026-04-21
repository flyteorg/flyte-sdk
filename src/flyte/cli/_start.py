import rich_click as click


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


_DEFAULT_DEMO_IMAGE = "ghcr.io/flyteorg/flyte-sandbox-v2:nightly"
_DEFAULT_DEMO_GPU_IMAGE = "ghcr.io/flyteorg/flyte-demo:gpu-latest"


@start.command()
@click.option(
    "--image",
    default=None,
    show_default=f"{_DEFAULT_DEMO_IMAGE} ({_DEFAULT_DEMO_GPU_IMAGE} when --gpu)",
    help="Docker image to use for the demo cluster.",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Enable dev mode inside the demo cluster (sets FLYTE_DEV=True).",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=False,
    help="Pass host GPUs into the demo container (adds --gpus all to docker run). "
    "Requires an NVIDIA-enabled host. Defaults --image to a GPU-capable image "
    "if --image is not explicitly set.",
)
@click.pass_context
def demo(ctx: click.Context, image: str | None, dev: bool, gpu: bool):
    """Start a local Flyte demo cluster."""
    from flyte.cli._demo import launch_demo

    if image is None:
        image = _DEFAULT_DEMO_GPU_IMAGE if gpu else _DEFAULT_DEMO_IMAGE

    log_format = getattr(ctx.obj, "log_format", "console") if ctx.obj else "console"
    launch_demo(image, dev, gpu=gpu, log_format=log_format)
