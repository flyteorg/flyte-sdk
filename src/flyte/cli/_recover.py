"""`flyte recover <run>` — create a new run from a prior run, reusing its succeeded actions.

Counterpart to `flyte run`, which launches *local* code: `recover` re-launches an *existing*
run — fetching its task + inputs from the platform, no local code needed — and reuses every
action that already succeeded, re-executing only what is needed to carry the run to
completion. Remote-only.

Recovery deliberately does *not* incorporate code changes: it replays the source run's task
and inputs as-is, providing durability against intermittent system- or network-level
failures. To replay a run with *new* code or inputs, use `flyte fork` (Union-only, available
via `pip install flyteplugins-union`).
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


@click.command("recover", cls=click.RichCommand)
@click.argument("run_name", required=True)
@click.option("-p", "--project", default=None, help="Project for the new run (defaults to config).")
@click.option("-d", "--domain", default=None, help="Domain for the new run (defaults to config).")
@click.option("--name", default=None, help="Name for the new run (a random name is generated if unset).")
@click.option("-e", "--env", "env", multiple=True, help="Env var KEY=VALUE for the new run. Repeatable.")
@click.option("--label", "label", multiple=True, help="Label KEY=VALUE for the new run. Repeatable.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream the parent action logs after launch.")
@click.option(
    "--force-replay-action",
    "force_replay_action",
    multiple=True,
    help="Name of an action to re-execute even though it succeeded in the source run. "
    "Repeatable. A listed parent re-enqueues its children (list them too to force the "
    "whole subtree); unknown names are ignored.",
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
def recover(
    ctx: click.Context,
    run_name: str,
    project: Optional[str],
    domain: Optional[str],
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    force_replay_action: Tuple[str, ...],
    allow_missing_outputs: bool,
) -> None:
    """Recover run RUN_NAME: reuse its succeeded actions, re-run only what failed or never ran.

    Fetches the prior run's task + inputs from the platform (no local code needed) and launches a
    new run that reuses everything the source run already completed. Code changes are *not*
    picked up — recovery is for carrying a run past intermittent infrastructure failures. Use
    `--force-replay-action` to re-execute specific actions anyway.

    Remote-only. To replay a run with new code or inputs, use `flyte fork`
    (`pip install flyteplugins-union`).

    Examples:

        $ flyte recover ul56wcvgqrb9vzhzz5l2
        $ flyte recover ul56wcvgqrb9vzhzz5l2 --name retry-1 --follow
        $ flyte recover ul56wcvgqrb9vzhzz5l2 --force-replay-action a3 --force-replay-action a7
    """
    config = common.initialize_config(ctx, project=project, domain=domain)
    asyncio.run(_execute(run_name, name, env, label, follow, force_replay_action, allow_missing_outputs, config))


async def _execute(
    run_name: str,
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    force_replay_action: Tuple[str, ...],
    allow_missing_outputs: bool,
    config: common.CLIConfig,
) -> None:
    import flyte
    from flyte._status import status

    console = common.get_console()
    try:
        status.step(f"Recovering {run_name}...")
        runner = flyte.with_runcontext(
            mode="remote",
            name=name,
            env_vars=_parse_kv(env, "--env"),
            labels=_parse_kv(label, "--label"),
            recover=True,
            recover_force_rerun_actions=force_replay_action or None,
            allow_missing_source_outputs=allow_missing_outputs,
        )
        result = await runner.rerun.aio(run_name)
    except Exception as e:
        console.print(f"[red]✕ Recovery failed:[/red] {e}")
        return

    if config.output_format in ("json", "table-simple"):
        run_info = f"Created Run: {result.name}"
    else:
        run_info = f"[green bold]Created Run: {result.name}[/green bold]"
    console.print(common.get_panel("Recover", run_info, config.output_format))
    common.print_url(console, result.url, of=config.output_format)

    if follow:
        status.step("Waiting for log stream...")
        await result.show_logs.aio(max_lines=30, show_ts=True, raw=False)
