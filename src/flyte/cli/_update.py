import rich_click as click

import flyte.remote as remote

from . import _common as common


@click.group(name="update")
def update():
    """
    Update various flyte entities.
    """


@update.command("project", cls=click.RichCommand)
@click.argument("id", type=str)
@click.option("--name", type=str, default=None, help="Update the project display name.")
@click.option("--description", type=str, default=None, help="Update the project description.")
@click.option(
    "--label",
    "-l",
    multiple=True,
    callback=common.key_value_callback,
    help="Set labels as key=value pairs. Can be specified multiple times. Replaces all existing labels.",
)
@click.option("--archive/--unarchive", default=None, help="Archive or unarchive the project.")
@click.pass_obj
def project(
    cfg: common.CLIConfig,
    id: str,
    name: str | None,
    description: str | None,
    label: dict[str, str] | None,
    archive: bool | None,
):
    """
    Update a project's name, description, labels, or archive state.

    \b
    Example usage:

    ```bash
    flyte update project my_project --archive
    flyte update project my_project --unarchive
    flyte update project my_project --description "New description"
    flyte update project my_project --name "New Display Name"
    flyte update project my_project --label team=ml --label env=prod
    ```
    """
    if name is None and description is None and label is None and archive is None:
        raise click.UsageError(
            "At least one of --name, --description, --label, or --archive/--unarchive must be provided."
        )

    cfg.init()
    console = common.get_console()

    state = None
    if archive is not None:
        state = "archived" if archive else "active"

    with console.status(f"Updating project {id}..."):
        remote.Project.update(id=id, name=name, description=description, labels=label, state=state)

    parts = []
    if name is not None:
        parts.append(f"name set to '{name}'")
    if description is not None:
        parts.append("description updated")
    if label is not None:
        parts.append("labels updated")
    if archive is True:
        parts.append("[red]archived[/red]")
    elif archive is False:
        parts.append("[green]unarchived[/green]")
    console.print(f"Project [bold]{id}[/bold]: {', '.join(parts)}.")


@update.command("trigger", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.option("--activate/--deactivate", required=True, help="Activate or deactivate the trigger.")
@click.pass_obj
def trigger(cfg: common.CLIConfig, name: str, task_name: str, activate: bool, project: str | None, domain: str | None):
    """
    Update a trigger.

    \b
    Example usage:

    ```bash
    flyte update trigger <trigger_name> <task_name> --activate | --deactivate
    [--project <project_name> --domain <domain_name>]
    ```
    """
    cfg.init(project, domain)
    console = common.get_console()
    to_state = "active" if activate else "deactivate"
    with console.status(f"Updating trigger {name} for task {task_name} to {to_state}..."):
        remote.Trigger.update(name, task_name, activate)
    console.print(f"Trigger updated and is set to [fuchsia]{to_state}[/fuchsia]")


@update.command("app", cls=common.CommandBase)
@click.argument("name", type=str)
@click.option("--activate/--deactivate", "is_activate", default=None, help="Activate or deactivate app.")
@click.option("--wait", is_flag=True, default=False, help="Wait for the app to reach the desired state.")
@click.pass_obj
def app(
    cfg: common.CLIConfig, name: str, is_activate: bool | None, wait: bool, project: str | None, domain: str | None
):
    """
    Update an app by starting or stopping it.

    \b
    Example usage:

    ```bash
    flyte update app <app_name> --activate | --deactivate [--wait] [--project <project_name>] [--domain <domain_name>]
    ```
    """

    if is_activate is None:
        raise click.UsageError("Missing option '--activate' / '--deactivate'.")

    cfg.init(project, domain)
    console = common.get_console()

    app_obj = remote.App.get(name=name)
    if is_activate:
        if app_obj.is_active():
            console.print(f"App [yellow]{name}[/yellow] is already active.")
            return
        state = "activate"
        color = "green"
        with console.status(f"Activating app {name}..."):
            app_obj.activate(wait=wait)
    else:
        state = "deactivate"
        color = "red"
        if app_obj.is_deactivated():
            console.print(f"App [red]{name}[/red] is already deactivated.")
            return
        with console.status(f"Deactivating app {name}..."):
            app_obj.deactivate(wait=wait)

    if wait:
        console.print(f"App [{color}]{name}[/{color}] {state}d successfully")
    else:
        console.print(f"App [{color}]{name}[/{color}] {state} initiated")
