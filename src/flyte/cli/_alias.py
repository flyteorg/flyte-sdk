import rich_click as click

from . import _common as common


@click.group(name="alias")
def alias():
    """
    Manage task aliases: mutable named pointers to immutable task versions.

    An alias such as `prod` is what external callers launch through, so they never
    need to know a version. Deploying a task does not move an alias; only
    `flyte alias set` does.
    """


@alias.command("set", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.option("--to", "version", type=str, required=True, help="Task version the alias should point at.")
@click.pass_obj
def set_alias(cfg: common.CLIConfig, name: str, task_name: str, version: str, project: str | None, domain: str | None):
    """
    Create an alias, or move it to a different version — promote, or roll back.

    The same command does both: pointing `prod` at an older version is a rollback and
    needs no rebuild or redeploy.

    \b
    Example usage:

    ```bash
    flyte alias set prod my_env.my_task --to v1.4.0
    [--project <project_name> --domain <domain_name>]
    ```
    """
    from flyte.remote import TaskAlias

    cfg.init(project, domain)
    console = common.get_console()
    with console.status(f"Pointing alias {name} at {version}..."):
        updated, previous = TaskAlias.set(task_name=task_name, alias=name, version=version)

    if previous:
        console.print(f"[bold]{name}[/bold]: {previous} [fuchsia]->[/fuchsia] {updated.version}")
    else:
        console.print(f"[bold]{name}[/bold] created [fuchsia]->[/fuchsia] {updated.version}")


@alias.command("get", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.pass_obj
def get_alias(cfg: common.CLIConfig, name: str, task_name: str, project: str | None, domain: str | None):
    """
    Show which version an alias currently resolves to, and who last moved it.
    """
    from flyte.remote import TaskAlias

    cfg.init(project, domain)
    console = common.get_console()
    found = TaskAlias.get(task_name=task_name, alias=name)
    console.print(common.format(f"Alias {name}", [found], cfg.output_format))


@alias.command("list", cls=common.CommandBase)
@click.argument("task_name", type=str)
@click.option("--limit", type=int, default=100, help="Limit the number of aliases to fetch.")
@click.pass_obj
def list_aliases(cfg: common.CLIConfig, task_name: str, limit: int, project: str | None, domain: str | None):
    """
    List every alias defined for a task.
    """
    from flyte.remote import TaskAlias

    cfg.init(project, domain)
    console = common.get_console()
    console.print(
        common.format(
            f"Aliases for {task_name}", TaskAlias.listall(task_name=task_name, limit=limit), cfg.output_format
        )
    )


@alias.command("history", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.option("--limit", type=int, default=100, help="Limit the number of revisions to fetch.")
@click.pass_obj
def alias_history(
    cfg: common.CLIConfig, name: str, task_name: str, limit: int, project: str | None, domain: str | None
):
    """
    Show every move of an alias — from which version to which, by whom, when.

    Newest first. This is the audit trail for promotions and rollbacks.
    """
    from flyte.remote import TaskAlias

    cfg.init(project, domain)
    console = common.get_console()
    console.print(
        common.format(
            f"History for {name}", TaskAlias.history(task_name=task_name, alias=name, limit=limit), cfg.output_format
        )
    )


@alias.command("delete", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.pass_obj
def delete_alias(cfg: common.CLIConfig, name: str, task_name: str, project: str | None, domain: str | None):
    """
    Remove an alias. The versions it pointed at are unaffected.
    """
    from flyte.remote import TaskAlias

    cfg.init(project, domain)
    console = common.get_console()
    with console.status(f"Deleting alias {name}..."):
        TaskAlias.delete(task_name=task_name, alias=name)
    console.print(f"Alias [bold]{name}[/bold] deleted.")
