import rich_click as click


@click.group()
def start():
    """Start various Flyte services."""
    pass


@start.command()
def tui():
    """Launch TUI explore mode to browse past local runs."""
    from flyte.cli._tui import launch_tui_explore

    launch_tui_explore()
