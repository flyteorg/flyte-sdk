"""`flyte rerun <run>` — re-run an existing run with its own code + exact inputs.

Counterpart to `flyte run`, which launches *local* code: `rerun` re-launches an *existing* run,
fetching its task + inputs from the platform, so no local code is needed.

* `flyte rerun <run>` creates a whole new run with the same inputs and re-executes the whole
  workflow again, subject to global caching.
* `flyte rerun <run> --recover` creates a whole new run with the same inputs but reuses the prior
  run's succeeded actions, re-executing only what failed or never ran.
* `flyte rerun <run> --action-name <action>` re-runs just that one action from the run: the new
  run is rooted at that action's task, executed with the exact inputs it received. Mutually
  exclusive with `--recover`.

Recovery deliberately reuses the source run's code and inputs as-is — it provides durability
against intermittent system- or network-level failures, not a way to patch a run.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

import rich_click as click

from . import _common as common


def _parse_kv(items: Tuple[str, ...], flag: str) -> Optional[Dict[str, str]]:
    """Parse repeated `KEY=VALUE` flag values into a dict (None if none given)."""
    if not items:
        return None
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"Invalid {flag} value {item!r}: expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise click.BadParameter(f"Invalid {flag} value {item!r}: key must not be empty.")
        parsed[key] = value
    return parsed


@click.command("rerun", cls=click.RichCommand)
@click.argument("run_name", required=True)
@click.option("-p", "--project", default=None, help="Project for the new run (defaults to config).")
@click.option("-d", "--domain", default=None, help="Domain for the new run (defaults to config).")
@click.option("--name", default=None, help="Name for the new run (a random name is generated if unset).")
@click.option("-e", "--env", "env", multiple=True, help="Env var KEY=VALUE for the new run. Repeatable.")
@click.option("--label", "label", multiple=True, help="Label KEY=VALUE for the new run. Repeatable.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream the parent action logs after launch.")
@click.option(
    "--recover",
    is_flag=True,
    default=False,
    help="Reuse the prior run's succeeded actions, re-running only what failed or never ran. Remote-only.",
)
@click.option(
    "--action-name",
    "action_name",
    default=None,
    help="Re-run only this action from the run, instead of the whole run: the new run is rooted "
    "at that action's task with the inputs it received. Cannot be combined with --recover. "
    "List names with `flyte get action <run>`.",
)
@click.option(
    "--force-rerun-action",
    "force_rerun_action",
    multiple=True,
    help="With --recover: name of an action to re-execute even though it succeeded in the "
    "source run. Repeatable. A listed parent re-enqueues its children (list them too to "
    "force the whole subtree); unknown names are ignored.",
)
@click.option(
    "--allow-missing-outputs",
    "allow_missing_outputs",
    is_flag=True,
    default=False,
    help="Proceed when the source run's outputs were cleaned up from storage, using its inputs "
    "URI directly. The inputs cannot be verified from the client — if they were deleted too, "
    "the new run fails at runtime.",
)
@click.pass_context
def rerun(
    ctx: click.Context,
    run_name: str,
    project: Optional[str],
    domain: Optional[str],
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    recover: bool,
    action_name: Optional[str],
    force_rerun_action: Tuple[str, ...],
    allow_missing_outputs: bool,
) -> None:
    """Re-run an existing run RUN_NAME with its original code and inputs.

    Fetches the prior run's task + inputs from the platform (no local code needed) and launches a
    new run that returns the same way `flyte run` does. `--recover` reuses the prior run's
    succeeded actions, re-running only what failed or never ran; `--force-rerun-action` forces
    named actions to re-execute anyway.

    `--action-name` narrows the whole thing to a single action: the new run is rooted at that
    action's task, run with the exact inputs it received inside RUN_NAME. That is always a plain
    re-execution, so it cannot be combined with `--recover`.

    Examples:

        $ flyte rerun rxyz
        $ flyte rerun rxyz --name retry-1 --follow
        $ flyte rerun rxyz --recover
        $ flyte rerun rxyz --recover --force-rerun-action a3 --force-rerun-action a7
        $ flyte rerun rxyz --action-name a3
    """
    if force_rerun_action and not recover:
        raise click.UsageError("--force-rerun-action requires --recover")
    if action_name and recover:
        raise click.UsageError(
            "--action-name cannot be combined with --recover: recovery matches succeeded actions "
            "from the source run by name, and a run rooted at a single action has a different "
            "action tree. Re-run the action on its own, or recover the whole run."
        )
    config = common.initialize_config(ctx, project=project, domain=domain)
    asyncio.run(
        _execute(
            run_name, name, env, label, follow, recover, action_name, force_rerun_action, allow_missing_outputs, config
        )
    )


async def _execute(
    run_name: str,
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    recover: bool,
    action_name: Optional[str],
    force_rerun_action: Tuple[str, ...],
    allow_missing_outputs: bool,
    config: common.CLIConfig,
) -> None:
    import flyte
    from flyte._status import status

    console = common.get_console()
    try:
        target = f"{run_name}/{action_name}" if action_name else run_name
        status.step(f"{'Recovering' if recover else 'Re-running'} {target}...")
        runner = flyte.with_runcontext(
            mode="remote",
            name=name,
            env_vars=_parse_kv(env, "--env"),
            labels=_parse_kv(label, "--label"),
            allow_missing_source_outputs=allow_missing_outputs,
        )
        result = await runner.rerun.aio(
            run_name,
            action_name=action_name or "a0",
            recover=recover,
            force_rerun_actions=force_rerun_action or None,
        )
    except Exception as e:
        console.print(f"[red]✕ {'Recovery' if recover else 'Re-run'} failed:[/red] {e}")
        return

    if config.output_format in ("json", "table-simple"):
        run_info = f"Created Run: {result.name}"
    else:
        run_info = f"[green bold]Created Run: {result.name}[/green bold]"
    console.print(common.get_panel("Recover" if recover else "Rerun", run_info, config.output_format))
    common.print_url(console, result.url, of=config.output_format)

    if follow:
        status.step("Waiting for log stream...")
        await result.show_logs.aio(max_lines=30, show_ts=True, raw=False)
