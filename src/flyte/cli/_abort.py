import rich_click as click

import flyte.remote as remote
from flyte.cli import _common as common


@click.group(name="abort")
def abort():
    """
    Abort an ongoing process.
    """


@abort.command(cls=common.CommandBase)
@click.argument("run-name", type=str, required=True)
@click.option("--reason", default="Manually aborted from the CLI", required=False, help="The reason to abort the run.")
@click.pass_obj
def run(cfg: common.CLIConfig, run_name: str, reason: str, project: str | None = None, domain: str | None = None):
    """
    Abort a run.
    """

    cfg.init(project=project, domain=domain)
    r = remote.Run.get(name=run_name)
    if r:
        console = common.get_console()
        with console.status(f"Aborting run '{run_name}'...", spinner="dots"):
            r.abort(reason=reason)
        console.print(f"Run '{run_name}' has been aborted.")


@abort.command(cls=common.CommandBase)
@click.argument("run-name", type=str, required=True)
@click.argument("action-name", type=str, required=True)
@click.option("--reason", default="Manually aborted from the CLI", required=False, help="The reason to abort the run.")
@click.pass_obj
def action(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str,
    reason: str,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Abort an action associated with a run.
    """

    cfg.init(project=project, domain=domain)

    a = remote.Action.get(run_name=run_name, name=action_name)
    if a:
        console = common.get_console()
        with console.status(f"Aborting action '{action_name}' for run '{run_name}'...", spinner="dots"):
            a.abort(reason=reason)
        console.print(f"Action '{action_name}' for run '{run_name}' has been aborted.")
