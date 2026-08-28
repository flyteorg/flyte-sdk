"""``flyte backfill <trigger>`` -- re-run the slots a scheduled trigger missed.

Expands the trigger's schedule across a window, shows exactly which runs it would
create, and on approval launches a driver run that creates them from inside the
cluster.

Slots that already ran are skipped, because a backfilled run is named the same way
a real scheduled fire is. ``--force`` re-runs them under salted names instead.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import rich_click as click

from . import _common as common


def _parse_time(value: str | None, *, what: str) -> datetime | None:
    """Accept an ISO timestamp, a plain date, or a relative age like ``30d`` / ``12h``."""
    if not value:
        return None
    raw = value.strip()
    if raw.lower() == "now":
        return datetime.now(timezone.utc)
    if len(raw) > 1 and raw[-1].lower() in "dhm" and raw[:-1].replace(".", "", 1).isdigit():
        amount = float(raw[:-1])
        unit = {"d": "days", "h": "hours", "m": "minutes"}[raw[-1].lower()]
        return datetime.now(timezone.utc) - timedelta(**{unit: amount})
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise click.BadParameter(
            f"{what} must be an ISO timestamp (2026-05-01T02:00), a date (2026-05-01), "
            f"or a relative age (30d, 12h). Got {value!r}."
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@click.command("backfill", cls=click.RichCommand)
@click.argument("trigger_name", required=True)
@click.option("--task-name", default=None, help="Task the trigger belongs to. Looked up if omitted.")
@click.option("-p", "--project", default=None, help="Project (defaults to config).")
@click.option("-d", "--domain", default=None, help="Domain (defaults to config).")
@click.option(
    "--from",
    "start",
    required=True,
    help="Start of the window: ISO timestamp, a date, or a relative age such as 30d.",
)
@click.option("--to", "end", default=None, help="End of the window. Defaults to now.")
@click.option("--queue", default=None, help="Queue for the driver run. Backfilled runs use the trigger's own queue.")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run slots that already ran. Names are salted so the new runs do not collide with the originals.",
)
@click.option("--suffix", default=None, help="With --force: extra salt distinguishing this backfill from earlier ones.")
@click.option(
    "--max-runs",
    default=None,
    type=int,
    help="Cap on runs created (default 100, matching the console).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Show the plan and exit without creating anything.")
@click.option("-y", "--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream the driver run's logs after launch.")
@click.pass_context
def backfill(
    ctx: click.Context,
    trigger_name: str,
    task_name: Optional[str],
    project: Optional[str],
    domain: Optional[str],
    start: str,
    end: Optional[str],
    queue: Optional[str],
    force: bool,
    suffix: Optional[str],
    max_runs: Optional[int],
    dry_run: bool,
    yes: bool,
    follow: bool,
) -> None:
    """Backfill the scheduled trigger TRIGGER_NAME over a window of time.

    Every slot the schedule would have fired in the window becomes a run, named the
    way a real fire is named. Slots that already ran are skipped, so running the
    same backfill twice is a no-op. ``--force`` re-runs them under salted names.

    Examples:

        $ flyte backfill nightly_eval --from 30d
        $ flyte backfill nightly_eval --from 2026-05-01 --to 2026-05-15 --dry-run
        $ flyte backfill nightly_eval --from 7d --force --suffix rerun-2
    """
    if suffix and not force:
        raise click.UsageError("--suffix requires --force")
    started = _parse_time(start, what="--from")
    ended = _parse_time(end, what="--to") or datetime.now(timezone.utc)
    if started is None or started >= ended:
        raise click.UsageError("--from must be earlier than --to")

    config = common.initialize_config(ctx, project=project, domain=domain)
    asyncio.run(
        _execute(
            trigger_name=trigger_name,
            task_name=task_name,
            start=started,
            end=ended,
            queue=queue,
            force=force,
            suffix=suffix,
            max_runs=max_runs,
            dry_run=dry_run,
            yes=yes,
            follow=follow,
            config=config,
        )
    )


async def _resolve_task_name(trigger_name: str, task_name: Optional[str]) -> str:
    """Find which task a trigger belongs to, when the caller did not say."""
    if task_name:
        return task_name
    from flyte.remote import Trigger

    matches = []
    async for trig in await Trigger.listall.aio():
        if trig.name == trigger_name:
            matches.append(trig)
    if not matches:
        raise click.ClickException(
            f"No trigger named {trigger_name!r} in this project/domain. Pass --task-name to disambiguate."
        )
    if len(matches) > 1:
        tasks = ", ".join(sorted({m.task_name for m in matches}))
        raise click.ClickException(
            f"Several tasks have a trigger named {trigger_name!r} ({tasks}). Pass --task-name to choose one."
        )
    return matches[0].task_name


def _render_plan(plan, console, output_format: str) -> None:
    from rich.table import Table

    header = (
        f"Trigger   {plan.trigger_name}  →  {plan.task_name}\n"
        f"Schedule  {plan.schedule}\n"
        f"Window    {plan.start.isoformat()}  →  {plan.end.isoformat()}\n"
        f"Force     {'on' + (f' (salt {plan.salt})' if plan.salt else '') if plan.force else 'off'}"
    )
    console.print(common.get_panel("Backfill", header, output_format))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Scheduled at")
    table.add_column("Run name")
    table.add_column("Action")
    shown = plan.slots[:20]
    for slot in shown:
        if plan.force and slot.already_ran:
            action = "[yellow]re-run (overrides existing)[/yellow]"
        elif slot.already_ran:
            action = "[dim]skip — already ran[/dim]"
        else:
            action = "[green]create[/green]"
        table.add_row(slot.scheduled_at.isoformat(), slot.run_name, action)
    console.print(table)
    if len(plan.slots) > len(shown):
        console.print(f"[dim]… and {len(plan.slots) - len(shown)} more[/dim]")

    console.print(f"\n[bold]{len(plan.to_create)}[/bold] run(s) to create, {len(plan.skipped)} skipped as already run.")
    if plan.truncated:
        console.print(
            f"[yellow]{plan.truncated} further slot(s) fall outside the {plan.max_runs}-run cap "
            f"and will not be created. Narrow the window or raise --max-runs.[/yellow]"
        )


async def _execute(
    *,
    trigger_name: str,
    task_name: Optional[str],
    start: datetime,
    end: datetime,
    queue: Optional[str],
    force: bool,
    suffix: Optional[str],
    max_runs: Optional[int],
    dry_run: bool,
    yes: bool,
    follow: bool,
    config: common.CLIConfig,
) -> None:
    from flyte._initialize import get_init_config
    from flyte.backfill import DEFAULT_MAX_RUNS, build_plan
    from flyte.backfill._driver import launch_backfill
    from flyte.backfill._execute import probe_existing
    from flyte.remote import Trigger

    console = common.get_console()
    resolved_task = await _resolve_task_name(trigger_name, task_name)
    details = await Trigger.get.aio(name=trigger_name, task_name=resolved_task)
    cfg = get_init_config()

    # Build once without existence info to learn the candidate names, probe those,
    # then rebuild so the displayed plan reflects what already ran.
    def _build(existing):
        return build_plan(
            details=details,
            task_name=resolved_task,
            org=cfg.org or "",
            project=cfg.project or "",
            domain=cfg.domain or "",
            start=start,
            end=end,
            force=force,
            suffix=suffix,
            queue=queue,
            max_runs=max_runs or DEFAULT_MAX_RUNS,
            existing=existing,
        )

    provisional = _build(None)
    if not provisional.slots:
        console.print("[yellow]No scheduled slots fall in that window — nothing to backfill.[/yellow]")
        return

    with common.cli_status(config.output_format, "Checking which slots already ran..."):
        candidates = [c for slot in provisional.slots for c in slot.candidates]
        existing = await probe_existing(candidates)
    plan = _build(existing)

    _render_plan(plan, console, config.output_format)

    if dry_run:
        console.print("\n[dim]Dry run — nothing was created.[/dim]")
        return
    if not plan.to_create:
        console.print("\n[dim]Every slot in this window has already run. Use --force to re-run them.[/dim]")
        return
    if not yes:
        click.confirm(f"\nCreate {len(plan.to_create)} run(s)?", abort=True)

    run = launch_backfill(plan, name=None)
    if config.output_format in ("json", "table-simple"):
        info = f"Backfill run: {run.name}\nURL: {run.url}"
    else:
        info = (
            f"[green bold]Backfill run: {run.name}[/green bold]\n"
            f"➡️  [blue bold][link={run.url}]{run.url}[/link][/blue bold]\n"
            f"[dim]Creating {len(plan.to_create)} run(s) from inside the cluster.[/dim]"
        )
    console.print(common.get_panel("Backfill", info, config.output_format))

    if follow:
        await run.show_logs.aio(max_lines=30, show_ts=True, raw=False)
