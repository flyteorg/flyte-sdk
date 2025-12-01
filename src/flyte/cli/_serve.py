
import click


@click.group("serve")
@click.pass_context
def serve(_: click.Context):
    """
    Start the specific service. For example:

    ```bash
    flyte serve connector
    ```
    """
