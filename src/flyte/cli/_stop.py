import rich_click as click


@click.group()
def stop():
    """Stop various Flyte services."""


@stop.command()
def demo():
    """Pause the local Flyte demo cluster without removing it."""
    from flyte.cli._demo import stop_demo

    stop_demo()
