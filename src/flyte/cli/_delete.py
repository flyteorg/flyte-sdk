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

    cfg.init(project, domain)
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
def demo(volume: bool):
    """
    Stop and remove the local Flyte demo cluster container.
    """
    console = common.get_console()
    result = subprocess.run(["docker", "stop", "flyte-demo"], capture_output=True, check=False)
    if result.returncode == 0:
        # The container is started with --rm, so wait for it to be fully removed
        subprocess.run(["docker", "wait", "flyte-demo"], capture_output=True, check=False)
        console.print("[green]Demo cluster stopped.[/green]")
    else:
        console.print("[yellow]Demo cluster is not running.[/yellow]")

    if volume:
        result = subprocess.run(["docker", "volume", "rm", "flyte-demo"], capture_output=True, check=False)
        if result.returncode == 0:
            console.print("[green]Docker volume 'flyte-demo' deleted.[/green]")
        else:
            console.print("[yellow]Docker volume 'flyte-demo' does not exist.[/yellow]")
