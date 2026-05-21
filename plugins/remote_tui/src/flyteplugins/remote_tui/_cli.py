"""``flyte start remote-tui`` CLI command."""

from __future__ import annotations

import rich_click as click


@click.command(name="remote-tui")
@click.option(
    "-c",
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the Flyte configuration file. Defaults to ~/.flyte/config.yaml.",
)
@click.option(
    "--poll-interval",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds between run detail refreshes while a run is active.",
)
def remote_tui_cmd(config_file: str | None, poll_interval: float) -> None:
    """
    Interactive TUI for a remote Flyte v2 cluster.

    Browse runs, actions, logs, tasks, apps, and triggers. Requires a Flyte config file
    (``flyte create config``) and ``pip install flyteplugins-remote-tui`` (or ``flyte[tui]``).
    """
    from flyteplugins.remote_tui import launch_remote_tui

    launch_remote_tui(config=config_file, poll_interval=poll_interval)
