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
    Secret.delete(name=name)


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
    Trigger.delete(name=name, task_name=task_name)
