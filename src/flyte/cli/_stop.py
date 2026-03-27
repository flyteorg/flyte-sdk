import rich_click as click

import flyte.cli._common as common


@click.group(name="stop")
def stop():
    """Stop various Flyte services."""


@stop.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.pass_obj
def app(cfg: common.CLIConfig, name: str, project: str | None = None, domain: str | None = None):
    """Stop (deactivate) a deployed Flyte app."""
    from flyte.remote import App

    cfg.init(project, domain)
    console = common.get_console()
    with console.status(f"Stopping app {name}..."):
        app_obj = App.get(name=name, project=project, domain=domain)
        app_obj.deactivate(wait=True)
    console.log(f"[green]Successfully stopped app {name}[/green]")
