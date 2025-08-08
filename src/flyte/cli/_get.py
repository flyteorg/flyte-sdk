import asyncio
import json
from dataclasses import asdict
from typing import Tuple, Union

import rich_click as click
from rich.console import Console
from rich.pretty import pretty_repr

import flyte.remote._action

from . import _common as common


OUTPUT_FORMATS = click.Choice(["table", "json"])


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
    from flyte.remote import Project

    cfg.init()

    console = Console()
    if name:
        console.print(pretty_repr(Project.get(name)))
    else:
        console.print(common.get_table("Projects", Project.listall(), simple=cfg.simple))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of tasks to show.")
@click.option("--output-format", type=OUTPUT_FORMATS, default="table", help="Output format.")
@click.pass_obj
def run(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 100,
    output_format: str = "table",
):
    """
    Get a list of all runs, or details of a specific run by name.

    The run details will include information about the run, its status, but only the root action will be shown.

    If you want to see the actions for a run, use `get action <run_name>`.
    """
    from flyte.remote import Run, RunDetails

    cfg.init(project=project, domain=domain)

    console = Console()
    if name:
        details = RunDetails.get(name=name)
        if output_format == "json":
            console.print(json.dumps(details.to_json_dict(), indent=2))
        else:
            console.print(pretty_repr(details))
    else:
        details = list(Run.listall(limit=limit))
        if output_format == "json":
            details_json = [d.to_json_dict() for d in details]
            console.print(json.dumps(details_json, indent=2))
        else:
            console.print(common.get_table("Runs", details, simple=cfg.simple))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.argument("version", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of tasks to show.")
@click.option("--output-format", type=OUTPUT_FORMATS, default="table", help="Output format.")
@click.pass_obj
def task(
    cfg: common.CLIConfig,
    name: str | None = None,
    version: str | None = None,
    limit: int = 100,
    output_format: str = "table",
    project: str | None = None,
    domain: str | None = None,
):
    """
    Retrieve a list of all tasks, or details of a specific task by name and version.

    Currently, both `name` and `version` are required to get a specific task.
    """
    from flyte.remote import Task

    cfg.init(project=project, domain=domain)

    console = Console()
    if name:
        if version:
            v = Task.get(name=name, version=version)
            if v is None:
                raise click.BadParameter(f"Task {name} not found.")
            t = v.fetch()
            if output_format == "json":
                console.print(json.dumps(t.to_json_dict(), indent=2))
            else:
                console.print(pretty_repr(t))
        else:
            tasks = Task.listall(by_task_name=name, limit=limit)
            if output_format == "json":
                tasks_json = [t.to_json_dict() for t in tasks]
                console.print(json.dumps(tasks_json, indent=2))
            else:
                console.print(common.get_table("Tasks", tasks, simple=cfg.simple))
    else:
        tasks = Task.listall(limit=limit)
        if output_format == "json":
            tasks_json = [t.to_json_dict() for t in tasks]
            console.print(json.dumps(tasks_json, indent=2))
        else:
            console.print(common.get_table("Tasks", tasks, simple=cfg.simple))


@get.command(cls=common.CommandBase)
@click.argument("run_name", type=str, required=True)
@click.argument("action_name", type=str, required=False)
@click.option("--output-format", type=OUTPUT_FORMATS, default="table", help="Output format.")
@click.pass_obj
def action(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str | None = None,
    output_format: str = "table",
    project: str | None = None,
    domain: str | None = None,
):
    """
    Get all actions for a run or details for a specific action.
    """

    cfg.init(project=project, domain=domain)

    console = Console()
    if action_name:
        action = flyte.remote._action.Action.get(run_name=run_name, name=action_name)
        if output_format == "json":
            console.print(json.dumps(action.to_json_dict(), indent=2))
        else:
            console.print(pretty_repr(action))
    else:
        # List all actions for the run
        actions = flyte.remote._action.Action.listall(for_run_name=run_name)
        if output_format == "json":
            actions_json = [a.to_json_dict() for a in actions]
            console.print(json.dumps(actions_json, indent=2))
        else:
            console.print(common.get_table(f"Actions for {run_name}", actions, simple=cfg.simple))


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
    import flyte.remote as remote

    cfg.init(project=project, domain=domain)

    async def _run_log_view(_obj):
        task = asyncio.create_task(
            _obj.show_logs(
                max_lines=lines, show_ts=show_ts, raw=not pretty, attempt=attempt, filter_system=filter_system
            )
        )
        try:
            await task
        except KeyboardInterrupt:
            task.cancel()

    if action_name:
        obj = flyte.remote._action.Action.get(run_name=run_name, name=action_name)
    else:
        obj = remote.Run.get(run_name)
    asyncio.run(_run_log_view(obj))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--output-format", type=OUTPUT_FORMATS, default="table", help="Output format.")
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str | None = None,
    output_format: str = "table",
    project: str | None = None,
    domain: str | None = None,
):
    """
    Get a list of all secrets, or details of a specific secret by name.
    """
    import flyte.remote as remote

    cfg.init(project=project, domain=domain)

    console = Console()
    if name:
        secret = remote.Secret.get(name)
        if output_format == "json":
            console.print(json.dumps(secret.to_json_dict(), indent=2))
        else:
            console.print(pretty_repr(secret))
    else:
        secrets = remote.Secret.listall()
        if output_format == "json":
            console.print(json.dumps([s.to_json_dict() for s in secrets], indent=2))
        else:
            console.print(common.get_table("Secrets", secrets, simple=cfg.simple))


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

    import flyte.remote as remote

    cfg.init(project=project, domain=domain)
    console = Console()
    if action_name:
        obj = flyte.remote._action.ActionDetails.get(run_name=run_name, name=action_name)
    else:
        obj = remote.RunDetails.get(run_name)

    async def _get_io(
        details: Union[remote.RunDetails, flyte.remote._action.ActionDetails],
    ) -> Tuple[flyte.remote._action.ActionInputs | None, flyte.remote._action.ActionOutputs | None | str]:
        if inputs_only or outputs_only:
            if inputs_only:
                return await details.inputs(), None
            elif outputs_only:
                return None, await details.outputs()
        inputs = await details.inputs()
        outputs: flyte.remote._action.ActionOutputs | None | str = None
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
            simple=cfg.simple,
        )
    )


@get.command(cls=click.RichCommand)
@click.pass_obj
def config(cfg: common.CLIConfig):
    """
    Shows the automatically detected configuration to connect with the remote backend.

    The configuration will include the endpoint, organization, and other settings that are used by the CLI.
    """
    console = Console()
    console.print(cfg)
