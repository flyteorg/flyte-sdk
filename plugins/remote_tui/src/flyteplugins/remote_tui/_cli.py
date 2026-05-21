"""``flyte start remote-tui`` CLI command."""

from __future__ import annotations

import rich_click as click


@click.command(name="remote-tui")
@click.option("--project", default=None, help="Flyte project (overrides config).")
@click.option("--domain", default=None, help="Flyte domain (overrides config).")
@click.option(
    "--poll-interval",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds between run detail refreshes while a run is active.",
)
def remote_tui_cmd(project: str | None, domain: str | None, poll_interval: float) -> None:
    """
    Interactive TUI for a remote Flyte v2 cluster.

    Browse runs, actions, logs, tasks, apps, and triggers. Requires ``flyte create config``
    and ``pip install flyteplugins-remote-tui`` (or ``flyte[tui]``).
    """
    from flyteplugins.remote_tui import launch_remote_tui

    launch_remote_tui(project=project, domain=domain, poll_interval=poll_interval)
