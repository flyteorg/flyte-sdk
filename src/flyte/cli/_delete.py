import subprocess

import rich_click as click

import flyte.cli._common as common


@click.group(name="delete")
def delete():
    """
    Remove resources from a Flyte deployment.
    """


@delete.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.pass_obj
def secret(cfg: common.CLIConfig, name: str, project: str | None = None, domain: str | None = None):
    """
    Delete a secret. The name of the secret is required.
    """
    from flyte.remote import Secret

    if project is None:
        project = ""
    if domain is None:
        domain = ""
    cfg.init(project=project, domain=domain)
    console = common.get_console()
    with console.status(f"Deleting secret {name}..."):
        Secret.delete(name=name)
    console.print(f"Successfully deleted secret {name}.")


@delete.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.argument("task-name", type=str, required=True)
@click.pass_obj
def trigger(cfg: common.CLIConfig, name: str, task_name: str, project: str | None = None, domain: str | None = None):
    """
    Delete a trigger. The name of the trigger is required.
    """
    from flyte.remote import Trigger

    cfg.init(project, domain)
    console = common.get_console()

    with console.status(f"Deleting trigger {name}..."):
        Trigger.delete(name=name, task_name=task_name, project=project, domain=domain)

    console.log(f"[green]Successfully deleted trigger {name}[/green]")


@delete.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.pass_obj
def app(cfg: common.CLIConfig, name: str, project: str | None = None, domain: str | None = None):
    """
    Delete apps from a Flyte deployment.
    """
    from flyte.remote import App

    cfg.init(project, domain)
    console = common.get_console()
    with console.status(f"Deleting app {name}..."):
        App.delete(name=name, project=project, domain=domain)

    console.log(f"[green]Successfully deleted app {name} [/green]")


@delete.command()
@click.option(
    "--volume",
    is_flag=True,
    default=False,
    help="Also delete the Docker volume used for persistent storage.",
)
def devbox(volume: bool):
    """
    Stop and remove the local Flyte devbox cluster container.
    """
    console = common.get_console()
    result = subprocess.run(["docker", "rm", "-f", "flyte-devbox"], capture_output=True, check=False)
    if result.returncode == 0:
        console.print("[green]Devbox cluster stopped.[/green]")
    else:
        console.print("[yellow]Devbox cluster is not running.[/yellow]")

    if volume:
        result = subprocess.run(["docker", "volume", "rm", "flyte-devbox"], capture_output=True, check=False)
        if result.returncode == 0:
            console.print("[green]Docker volume 'flyte-devbox' deleted.[/green]")
        else:
            console.print("[yellow]Docker volume 'flyte-devbox' does not exist.[/yellow]")
