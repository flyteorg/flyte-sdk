import rich_click as click

import flyte.remote as remote

from . import _common as common


@click.group(name="update")
def update():
    """
    Update various flyte entities.
    """


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
