import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Tuple, Union

import rich_click as click
from rich.pretty import pretty_repr

import flyte.remote as remote
from flyte.models import ActionPhase
from flyte.remote._common import TimeFilter

from . import _common as common
from . import _params
from ._option import MutuallyExclusiveOption


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
@click.option("--archived", is_flag=True, default=False, help="Show archived projects instead of active ones.")
@click.pass_obj
def project(cfg: common.CLIConfig, name: str | None = None, archived: bool = False):
    """
    Get a list of all projects, or details of a specific project by name.

    By default, only active (unarchived) projects are shown. Use `--archived` to
    show archived projects instead.
    """
    cfg.init()

    console = common.get_console()
    if name:
        console.print(pretty_repr(remote.Project.get(name)))
    else:
        console.print(common.format("Projects", remote.Project.listall(archived=archived), cfg.output_format))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of runs to fetch when listing.")
@click.option(
    "--in-phase",  # multiple=True, TODO support multiple phases once values in works
    type=click.Choice([p.value for p in ActionPhase], case_sensitive=False),
    help="Filter runs by their status.",
)
@click.option("--only-mine", is_flag=True, default=False, help="Show only runs created by the current user (you).")
@click.option("--task-name", type=str, default=None, help="Filter runs by task name.")
@click.option("--task-version", type=str, default=None, help="Filter runs by task version.")
@click.option(
    "--created-after",
    type=_params.DateTimeType(),
    default=None,
    help="Show runs created at or after this datetime (UTC). Accepts ISO dates, 'now', 'today', or 'now - 1 day'.",
)
@click.option(
    "--created-before", type=_params.DateTimeType(), default=None, help="Show runs created before this datetime (UTC)."
)
@click.option(
    "--updated-after",
    type=_params.DateTimeType(),
    default=None,
    help="Show runs updated at or after this datetime (UTC). Accepts ISO dates, 'now', 'today', or 'now - 1 day'.",
)
@click.option(
    "--updated-before", type=_params.DateTimeType(), default=None, help="Show runs updated before this datetime (UTC)."
)
@click.pass_obj
def run(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 100,
    in_phase: str | Tuple[str, ...] | None = None,
    only_mine: bool = False,
    task_name: str | None = None,
    task_version: str | None = None,
    created_after: dt.datetime | None = None,
    created_before: dt.datetime | None = None,
    updated_after: dt.datetime | None = None,
    updated_before: dt.datetime | None = None,
):
    """
    Get a list of all runs, or details of a specific run by name.

    The run details will include information about the run, its status, but only the root action will be shown.

    If you want to see the actions for a run, use `get action <run_name>`.

    You can filter runs by task name and optionally task version:

    ```bash
    $ flyte get run --task-name my_task
    $ flyte get run --task-name my_task --task-version v1.0
    ```
    """

    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        details = remote.RunDetails.get(name=name)
        console.print(common.format(f"Run {name}", [details], "json"))
    else:
        if in_phase and isinstance(in_phase, str):
            in_phase = (ActionPhase(in_phase),)

        subject = None
        if only_mine:
            usr = remote.User.get()
            subject = usr.subject()

        def _utc(d: dt.datetime | None) -> dt.datetime | None:
            return d.replace(tzinfo=dt.timezone.utc) if d is not None and d.tzinfo is None else d

        created_at = (
            TimeFilter(after=_utc(created_after), before=_utc(created_before))
            if created_after or created_before
            else None
        )
        updated_at = (
            TimeFilter(after=_utc(updated_after), before=_utc(updated_before))
            if updated_after or updated_before
            else None
        )

        console.print(
            common.format(
                "Runs",
                remote.Run.listall(
                    limit=limit,
                    in_phase=in_phase,
                    created_by_subject=subject,
                    task_name=task_name,
                    task_version=task_version,
                    created_at=created_at,
                    updated_at=updated_at,
                ),
                cfg.output_format,
            )
        )


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.argument("version", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of tasks to fetch.")
@click.option("--entrypoint", is_flag=True, default=False, help="Show only entrypoint tasks.")
@click.pass_obj
def task(
    cfg: common.CLIConfig,
    name: str | None = None,
    limit: int = 100,
    version: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    entrypoint: bool = False,
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
                common.format(
                    "Tasks",
                    remote.Task.listall(by_task_name=name, limit=limit, entrypoint=entrypoint or None),
                    cfg.output_format,
                )
            )
    else:
        console.print(
            common.format("Tasks", remote.Task.listall(limit=limit, entrypoint=entrypoint or None), cfg.output_format)
        )


@get.command(cls=common.CommandBase)
@click.argument("run_name", type=str, required=True)
@click.argument("action_name", type=str, required=False)
@click.option(
    "--in-phase",
    type=click.Choice([p.value for p in ActionPhase], case_sensitive=False),
    help="Filter actions by their phase.",
)
@click.pass_obj
def action(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str | None = None,
    in_phase: str | None = None,
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
        if in_phase:
            in_phase_tuple = (ActionPhase(in_phase),)
        else:
            in_phase_tuple = None

        console.print(
            common.format(
                f"Actions for {run_name}",
                remote.Action.listall(for_run_name=run_name, in_phase=in_phase_tuple),
                cfg.output_format,
            )
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

    obj: Union[remote.Action, remote.Run]
    if action_name:
        obj = remote.Action.get(run_name=run_name, name=action_name)
    else:
        obj = remote.Run.get(name=run_name)
    asyncio.run(_run_log_view(obj))


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option(
    "--cluster-pool",
    type=str,
    default=None,
    help="Scope the secret to a cluster pool. Mutually exclusive with --project and --domain.",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["project", "domain"],
)
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str | None = None,
    cluster_pool: str | None = None,
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

    if cluster_pool and (project != "" or domain != ""):
        raise click.ClickException("Project and domain must not be set when --cluster-pool is specified.")

    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        console.print(common.format("Secret", [remote.Secret.get(name, cluster_pool=cluster_pool)], "json"))
    else:
        console.print(common.format("Secrets", remote.Secret.listall(cluster_pool=cluster_pool), cfg.output_format))


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
    obj: Union[remote.ActionDetails, remote.RunDetails]
    if action_name:
        obj = remote.ActionDetails.get(run_name=run_name, name=action_name)
    else:
        obj = remote.RunDetails.get(name=run_name)

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


@get.command(cls=common.CommandBase)
@click.option(
    "--to-file",
    "-o",
    "to_file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the scope's YAML to this file instead of printing it. The file "
    "round-trips through `flyte edit settings --from-file`.",
)
@click.pass_obj
def settings(
    cfg: common.CLIConfig,
    project: str | None = None,
    domain: str | None = None,
    to_file: Path | None = None,
):
    """
    Get settings for a scope as editable YAML.

    Renders three sections:

    \b
    * Local overrides — uncommented, applied at this scope.
    * Inherited settings — commented, with the scope they come from.
    * Available settings — commented placeholders for every key that
      isn't set anywhere yet, so you can see what can be configured.

    \b
    Examples:

    ```bash
    # Get ORG-level settings
    flyte get settings

    # Get settings for a domain
    flyte get settings --domain production

    # Get settings for a project (inherits from domain, which inherits from org)
    flyte get settings --domain production --project ml-pipeline

    # Dump to a file, edit it, then apply non-interactively
    flyte get settings --domain production -o prod.yaml
    # ...edit prod.yaml...
    flyte edit settings --domain production --from-file prod.yaml
    ```

    Use `flyte edit settings` to interactively modify these values.
    """
    from rich.panel import Panel

    cfg.init()

    console = common.get_console()
    s = remote.Settings.get_settings_for_edit(domain=domain, project=project)

    if to_file is not None:
        # Dump the raw YAML (with ### / ## / # markers preserved) so the file
        # round-trips through `flyte edit settings --from-file`.
        to_file.write_text(s.to_yaml() + "\n")
        console.print(
            f"[green]✓ Wrote settings for {s.scope_description()} (v{s._version}) to [bold]{to_file}[/bold][/green]"
        )
        return

    console.print(
        Panel(
            _stylize_settings_yaml(s.to_yaml()),
            title=f"[bold]Settings[/bold] · [cyan]{s.scope_description()}[/cyan] · [dim]v{s._version}[/dim]",
            title_align="left",
            border_style="bright_black",
            padding=(1, 2),
        )
    )


def _stylize_settings_yaml(yaml_content: str) -> "Any":
    """Render settings YAML for display, replacing comment markers with visual
    cues. The raw ``#`` / ``##`` / ``###`` prefixes emitted by
    ``Settings.to_yaml`` are stripped for readability — callers that need the
    round-trippable form (``flyte edit settings``) should use ``to_yaml``
    directly, *not* this function.

    Visual hierarchy:

    * ``### Section`` → ``▌ Section`` in bold bright cyan.
    * ``## description`` → the description text only, grey50.
    * ``# key: value`` → ``key: value`` rendered dim (clearly inactive but
      still legible so users can see what to uncomment). Any trailing
      ``  ## meta`` is lifted into a parenthesised italic suffix.
    * ``key: value`` → bold bright_blue key + bright_green value.
    """
    from rich.text import Text

    out = Text(no_wrap=False)
    lines = yaml_content.split("\n")

    def _append_kv(indent: str, key: str, value: str, key_style: str, val_style: str) -> None:
        out.append(indent)
        out.append(key, style=key_style)
        out.append(":", style="white")
        out.append(value, style=val_style)

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("### "):
            out.append(indent)
            out.append("▌ ", style="bold bright_cyan")
            out.append(stripped[4:], style="bold bright_cyan")
        elif stripped.startswith("## "):
            out.append(indent)
            out.append(stripped[3:], style="grey50")
        elif stripped.startswith("# "):
            content = stripped[2:]
            meta = ""
            meta_idx = content.find("  ## ")
            if meta_idx >= 0:
                meta = content[meta_idx + len("  ## ") :]
                content = content[:meta_idx]
            if ":" in content:
                key, value = content.split(":", 1)
                _append_kv(indent, key, value, key_style="magenta", val_style="grey66")
            else:
                out.append(indent)
                out.append(content, style="grey66")
            if meta:
                out.append(f"  ({meta})", style="italic grey50")
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
            _append_kv(indent, key, value, key_style="bold bright_magenta", val_style="bright_green")
        else:
            out.append(line)

        if i < len(lines) - 1:
            out.append("\n")
    return out


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


@get.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--limit", type=int, default=100, help="Limit the number of apps to fetch when listing.")
@click.option("--only-mine", is_flag=True, default=False, help="Show only apps created by the current user (you).")
@click.pass_obj
def app(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 100,
    only_mine: bool = False,
):
    """
    Get a list of all apps, or details of a specific app by name.

    Apps are long-running services deployed on the Flyte platform.
    """
    cfg.init(project=project, domain=domain)

    console = common.get_console()
    if name:
        app_details = remote.App.get(name=name)
        console.print(common.format(f"App {name}", [app_details], "json"))
    else:
        subject = None
        if only_mine:
            usr = remote.User.get()
            subject = usr.subject()

        console.print(
            common.format(
                "Apps",
                remote.App.listall(limit=limit, created_by_subject=subject),
                cfg.output_format,
            )
        )
