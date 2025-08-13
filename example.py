import rich_click as click
from rich.console import Console

console = Console()


@click.command()
def cli():
    url = (
        "https://example.com/this/is/a/really/long/url/"
        "that/will/be/wrapped/over/multiple/lines/"
        "but/still/needs/to/be/clickable"
    )

    console.print(f"[link={url}]{url}[/link]")


if __name__ == "__main__":
    cli()
