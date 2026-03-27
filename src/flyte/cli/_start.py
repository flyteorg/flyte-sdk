import rich_click as click

import flyte.cli._common as common


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


@start.command()
@click.option(
    "--image",
    default="ghcr.io/flyteorg/flyte-sandbox-v2:nightly",
    show_default=True,
    help="Docker image to use for the demo cluster.",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Enable dev mode inside the demo cluster (sets FLYTE_DEV=True).",
)
def demo(image: str, dev: bool):
    """Start a local Flyte demo cluster."""
    from flyte.cli._demo import launch_demo

    launch_demo(image, dev)


@start.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.pass_obj
def app(cfg: common.CLIConfig, name: str, project: str | None = None, domain: str | None = None):
    """Start (activate) a deployed Flyte app."""
    from flyte.remote import App

    cfg.init(project, domain)
    console = common.get_console()
    with console.status(f"Starting app {name}..."):
        app_obj = App.get(name=name, project=project, domain=domain)
        app_obj.activate(wait=True)
    console.log(f"[green]Successfully started app {name}[/green]")
