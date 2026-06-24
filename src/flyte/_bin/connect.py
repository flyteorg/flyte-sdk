from typing import List

import click


@click.group()
def _connect():
    """Debug commands for Flyte."""


@_connect.command(name="c0")
@click.option(
    "--port",
    default="8000",
    is_flag=False,
    type=int,
    help="gRPC port for the connector service. Defaults to 8000",
)
@click.option(
    "--connect_port",
    default="8090",
    is_flag=False,
    type=int,
    help="Connect (HTTP/1.1) port for the connector service. Defaults to 8090",
)
@click.option(
    "--prometheus_port",
    default="9090",
    is_flag=False,
    type=int,
    help="Prometheus port for the connector service. Defaults to 9090",
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
    help="It will wait for the specified number of seconds before shutting down the connector servers. It should only "
    "be used for testing.",
)
@click.option(
    "--modules",
    required=False,
    multiple=True,
    type=str,
    help="List of additional files or module that defines the connector",
)
@click.pass_context
def main(
    _: click.Context,
    port: int,
    connect_port: int,
    prometheus_port: int,
    worker: int,
    timeout: int | None,
    modules: List[str] | None,
):
    """
    Start the connector service, serving both gRPC (``--port``) and Connect/HTTP1.1 (``--connect_port``).
    """
    from flyte.connectors import ConnectorService

    ConnectorService.run(port, connect_port, prometheus_port, worker, timeout, modules)


if __name__ == "__main__":
    _connect()
