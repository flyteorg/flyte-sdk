import importlib
import os
import sys
from concurrent import futures

import grpc
import rich_click as click
from flyteidl2.service import connector_pb2
from flyteidl2.service.connector_pb2_grpc import (
    add_AsyncConnectorServiceServicer_to_server,
    add_ConnectorMetadataServiceServicer_to_server,
)
from rich.console import Console
from rich.table import Table


@click.group("serve")
@click.pass_context
def serve(ctx: click.Context):
    """
    Start the specific service. For example:

    ```bash
    flyte serve connector
    ```
    """
    pass


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
def connector(_: click.Context, port, prometheus_port, worker, timeout, modules):
    """
    Start a grpc server for the connector service.
    """
    import asyncio

    working_dir = os.getcwd()
    if all(os.path.realpath(path) != working_dir for path in sys.path):
        sys.path.append(working_dir)
    for m in modules:
        importlib.import_module(m)

    asyncio.run(_start_grpc_server(port, prometheus_port, worker, timeout))


async def _start_grpc_server(port: int, prometheus_port: int, worker: int, timeout: int):
    try:
        from flyte.connectors._server import (
            AsyncConnectorService,
            ConnectorMetadataService,
            SyncConnectorService,
        )
    except ImportError as e:
        raise ImportError(
            f"Flyte connector dependencies are not installed. Please install it using `pip install flyte[connector]`"
        ) from e

    click.secho(f"🚀 Starting the connector service...")
    _start_http_server(prometheus_port)

    print_metadata()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=worker))

    add_AsyncConnectorServiceServicer_to_server(AsyncConnectorService(), server)
    add_ConnectorMetadataServiceServicer_to_server(ConnectorMetadataService(), server)
    _start_health_check_server(server, worker)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    await server.wait_for_termination(timeout)


def _start_http_server(prometheus_port: int):
    try:
        from prometheus_client import start_http_server

        click.secho("Starting up the server to expose the prometheus metrics...")
        start_http_server(prometheus_port)
    except ImportError as e:
        click.secho(f"Failed to start the prometheus server with error {e}", fg="red")


def _start_health_check_server(server: grpc.Server, worker: int):
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        health_servicer = health.HealthServicer(
            experimental_non_blocking=True,
            experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=worker),
        )

        for service in connector_pb2.DESCRIPTOR.services_by_name.values():
            health_servicer.set(service.full_name, health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(health.SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)

        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    except ImportError as e:
        click.secho(f"Failed to start the health check servicer with error {e}", fg="red")


def print_metadata():
    from flyte.connectors import ConnectorRegistry

    connectors = ConnectorRegistry.list_connectors()

    table = Table(title="Connector Metadata")
    table.add_column("Connector Name", style="cyan", no_wrap=True)
    table.add_column("Support Task Types", style="cyan")

    for connector in connectors:
        categories = ""
        for category in connector.supported_task_categories:
            categories += f"{category.name} ({category.version}) "
        table.add_row(connector.name, categories)

    console = Console()
    console.print(table)
