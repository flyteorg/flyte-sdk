import asyncio
from typing import Tuple, Union, get_args

import rich_click as click
from rich.pretty import pretty_repr

import flyte.remote as remote

from . import _common as common


@click.group(name="get")
def get():
    """
    Retrieve resources from a Flyte deployment.

    You can get information about projects, runs, tasks, actions, secrets, logs and input/output values.

    Each command supports optional parameters to filter or specify the resource you want to retrieve.

    Using a `get` subcommand without any arguments will retrieve a list of available resources to get.
    For example:

    * `get project` (without specifying a project), will list all projects.
    * `get project my_project` will return the details of the project named `my_project`.

    In some cases, a partially specified command will act as a filter and return available further parameters.
    For example:

    * `get action my_run` will return all actions for the run named `my_run`.
    * `get action my_run my_action` will return the details of the action named `my_action` for the run `my_run`.
    """


@get.command()
@click.argument("name", type=str, required=False)
@click.pass_obj
def project(cfg: common.CLIConfig, name: str | None = None):
    """
    Get a list of all projects, or details of a specific project by name.
    """
    cfg.init()

    console = common.get_console()
    if name:
        console.print(pretty_repr(remote.Project.get(name)))
    else:
        console.print(common.format("Projects", remote.Project.listall(), cfg.output_format))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of runs to fetch when listing.")
@click.option(
    "--in-phase",  # multiple=True, TODO support multiple phases once values in works
    type=click.Choice(get_args(remote.Phase), case_sensitive=False),
    help="Filter runs by their status.",
)
@click.option("--only-mine", is_flag=True, default=False, help="Show only runs created by the current user (you).")
@click.pass_obj
def run(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 100,
    in_phase: str | Tuple[str, ...] | None = None,
    only_mine: bool = False,
):
    """
    Get a list of all runs, or details of a specific run by name.

    The run details will include information about the run, its status, but only the root action will be shown.

    If you want to see the actions for a run, use `get action <run_name>`.
    """

    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        details = remote.RunDetails.get(name=name)
        console.print(common.format(f"Run {name}", [details], "json"))
    else:
        if in_phase and isinstance(in_phase, str):
            in_phase = (in_phase,)

        subject = None
        if only_mine:
            usr = remote.User.get()
            subject = usr.subject()

        console.print(
            common.format(
                "Runs",
                remote.Run.listall(limit=limit, in_phase=in_phase, created_by_subject=subject),
                cfg.output_format,
            )
        )


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.argument("version", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of tasks to fetch.")
@click.pass_obj
def task(
    cfg: common.CLIConfig,
    name: str | None = None,
    limit: int = 100,
    version: str | None = None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Retrieve a list of all tasks, or details of a specific task by name and version.

    Currently, both `name` and `version` are required to get a specific task.
    """
    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        if version:
            v = remote.Task.get(name=name, version=version)
            if v is None:
                raise click.BadParameter(f"Task {name} not found.")
            t = v.fetch()
            console.print(common.format(f"Task {name}", [t], "json"))
        else:
            console.print(
                common.format("Tasks", remote.Task.listall(by_task_name=name, limit=limit), cfg.output_format)
            )
    else:
        console.print(common.format("Tasks", remote.Task.listall(limit=limit), cfg.output_format))


@get.command(cls=common.CommandBase)
@click.argument("run_name", type=str, required=True)
@click.argument("action_name", type=str, required=False)
@click.pass_obj
def action(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Get all actions for a run or details for a specific action.
    """
    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if action_name:
        console.print(
            common.format(
                f"Action {run_name}.{action_name}", [remote.Action.get(run_name=run_name, name=action_name)], "json"
            )
        )
    else:
        # List all actions for the run
        console.print(
            common.format(f"Actions for {run_name}", remote.Action.listall(for_run_name=run_name), cfg.output_format)
        )


@get.command(cls=common.CommandBase)
@click.argument("run_name", type=str, required=True)
@click.argument("action_name", type=str, required=False)
@click.option("--lines", "-l", type=int, default=30, help="Number of lines to show, only useful for --pretty")
@click.option("--show-ts", is_flag=True, help="Show timestamps")
@click.option(
    "--pretty",
    is_flag=True,
    default=False,
    help="Show logs in an auto-scrolling box, where number of lines is limited to `--lines`",
)
@click.option(
    "--attempt", "-a", type=int, default=None, help="Attempt number to show logs for, defaults to the latest attempt."
)
@click.option("--filter-system", is_flag=True, default=False, help="Filter all system logs from the output.")
@click.pass_obj
def logs(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    lines: int = 30,
    show_ts: bool = False,
    pretty: bool = True,
    attempt: int | None = None,
    filter_system: bool = False,
):
    """
    Stream logs for the provided run or action.
    If only the run is provided, only the logs for the parent action will be streamed:

    ```bash
    $ flyte get logs my_run
    ```

    If you want to see the logs for a specific action, you can provide the action name as well:

    ```bash
    $ flyte get logs my_run my_action
    ```

    By default, logs will be shown in the raw format and will scroll the terminal.
    If automatic scrolling and only tailing `--lines` number of lines is desired, use the `--pretty` flag:

    ```bash
    $ flyte get logs my_run my_action --pretty --lines 50
    ```
    """
    cfg.init(project=project, domain=domain)

    async def _run_log_view(_obj):
        task = asyncio.create_task(
            _obj.show_logs.aio(
                max_lines=lines, show_ts=show_ts, raw=not pretty, attempt=attempt, filter_system=filter_system
            )
        )
        try:
            await task
        except KeyboardInterrupt:
            task.cancel()

    if action_name:
        obj = remote.Action.get(run_name=run_name, name=action_name)
    else:
        obj = remote.Run.get(run_name)
    asyncio.run(_run_log_view(obj))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Get a list of all secrets, or details of a specific secret by name.
    """
    if project is None:
        project = ""
    if domain is None:
        domain = ""
    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        console.print(common.format("Secret", [remote.Secret.get(name)], "json"))
    else:
        console.print(common.format("Secrets", remote.Secret.listall(), cfg.output_format))


@get.command(cls=common.CommandBase)
@click.argument("run_name", type=str, required=True)
@click.argument("action_name", type=str, required=False)
@click.option("--inputs-only", "-i", is_flag=True, help="Show only inputs")
@click.option("--outputs-only", "-o", is_flag=True, help="Show only outputs")
@click.pass_obj
def io(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    inputs_only: bool = False,
    outputs_only: bool = False,
):
    """
    Get the inputs and outputs of a run or action.
    If only the run name is provided, it will show the inputs and outputs of the root action of that run.
    If an action name is provided, it will show the inputs and outputs for that action.
    If `--inputs-only` or `--outputs-only` is specified, it will only show the inputs or outputs respectively.

    Examples:

    ```bash
    $ flyte get io my_run
    ```

    ```bash
    $ flyte get io my_run my_action
    ```
    """
    if inputs_only and outputs_only:
        raise click.BadParameter("Cannot use both --inputs-only and --outputs-only")

    cfg.init(project=project, domain=domain)
    console = common.get_console()
    if action_name:
        obj = remote.ActionDetails.get(run_name=run_name, name=action_name)
    else:
        obj = remote.RunDetails.get(run_name)

    async def _get_io(
        details: Union[remote.RunDetails, remote.ActionDetails],
    ) -> Tuple[remote.ActionInputs | None, remote.ActionOutputs | None | str]:
        if inputs_only or outputs_only:
            if inputs_only:
                return await details.inputs(), None
            elif outputs_only:
                return None, await details.outputs()
        inputs = await details.inputs()
        outputs: remote.ActionOutputs | None | str = None
        try:
            outputs = await details.outputs()
        except Exception:
            # If the outputs are not available, we can still show the inputs
            outputs = "[red]not yet available[/red]"
        return inputs, outputs

    inputs, outputs = asyncio.run(_get_io(obj))
    # Show inputs and outputs side by side
    console.print(
        common.get_panel(
            "Inputs & Outputs",
            f"[green bold]Inputs[/green bold]\n{inputs}\n\n[blue bold]Outputs[/blue bold]\n{outputs}",
            cfg.output_format,
        )
    )


@get.command(cls=click.RichCommand)
@click.pass_obj
def config(cfg: common.CLIConfig):
    """
    Shows the automatically detected configuration to connect with the remote backend.

    The configuration will include the endpoint, organization, and other settings that are used by the CLI.
    """
    console = common.get_console()
    console.print(cfg)


@get.command(cls=common.CommandBase)
@click.argument("task_name", type=str, required=False)
@click.argument("name", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of triggers to fetch.")
@click.pass_obj
def trigger(
    cfg: common.CLIConfig,
    task_name: str | None = None,
    name: str | None = None,
    limit: int = 100,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Get a list of all triggers, or details of a specific trigger by name.
    """
    if name and not task_name:
        raise click.BadParameter("If you provide a trigger name, you must also provide the task name.")

    from flyte.remote import Trigger

    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        console.print(pretty_repr(Trigger.get(name=name, task_name=task_name)))
    else:
        console.print(common.format("Triggers", Trigger.listall(task_name=task_name, limit=limit), cfg.output_format))
