import rich_click as click

from flyte.cli import _common as common
import flyte.remote as remote


@click.group(name="abort")
def abort():
    """
    Abort an ongoing process.
    """


@abort.command(cls=common.CommandBase)
@click.argument("run-name", type=str, required=True)
@click.pass_obj
def run(cfg: common.CLIConfig, run_name: str, project: str | None = None, domain: str | None = None):
    """
    Abort a run.
    """
    from flyte.remote import Run

    cfg.init(project=project, domain=domain)
    r = Run.get(name=run_name)
    if r:
        console = common.get_console()
        with console.status(f"Aborting run '{run_name}'...", spinner="dots"):
            r.abort()
        console.print(f"Run '{run_name}' has been aborted.")

@abort.command(cls=common.CommandBase)
@click.argument("run-name", type=str, required=True)
@click.argument("action-name", type=str, required=True)
@click.pass_obj
def action(cfg: common.CLIConfig, run_name: str, action_name: str, project: str | None = None, domain: str | None = None):
    """
    Abort a run.
    """
    from flyte.remote import Run

    cfg.init(project=project, domain=domain)

    a = remote.Action.get(run_name=run_name, name=action_name)
    if a:
        console = common.get_console()
        with console.status(f"Aborting action 'action_name' for a '{run_name}'...", spinner="dots"):
            a.abort()
        console.print(f"Action '{action_name}' for a '{run_name}' has been aborted.")

