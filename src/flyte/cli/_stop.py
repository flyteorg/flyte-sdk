import rich_click as click


@click.group()
def stop():
    """Stop various Flyte services."""


@stop.command()
def devbox():
    """Pause the local Flyte devbox cluster without removing it."""
    from flyte.cli._devbox import stop_devbox

    stop_devbox()
