from typing import List

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


@serve.command()
@click.option(
    "--port",
    default="8000",
    is_flag=False,
    type=int,
    help="Grpc port for the connector service",
)
@click.option(
    "--prometheus_port",
    default="9090",
    is_flag=False,
    type=int,
    help="Prometheus port for the connector service",
)
@click.option(
    "--worker",
    default="10",
    is_flag=False,
    type=int,
    help="Number of workers for the grpc server",
)
@click.option(
    "--timeout",
    default=None,
    is_flag=False,
    type=int,
    help="It will wait for the specified number of seconds before shutting down grpc server. It should only be used "
    "for testing.",
)
@click.option(
    "--modules",
    required=False,
    multiple=True,
    type=str,
    help="List of additional files or module that defines the connector",
)
@click.pass_context
def connector(
    _: click.Context, port: int, prometheus_port: int, worker: int, timeout: int | None, modules: List[str] | None
):
    """
    Start a grpc server for the connector service.
    """
    from flyte.connectors import ConnectorService

    ConnectorService.run(port, prometheus_port, worker, timeout, modules)
