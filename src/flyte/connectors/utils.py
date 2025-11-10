import importlib
from concurrent import futures
from importlib.metadata import entry_points
from typing import List

import click
import grpc
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.service import connector_pb2
from flyteidl2.service.connector_pb2_grpc import (
    add_AsyncConnectorServiceServicer_to_server,
    add_ConnectorMetadataServiceServicer_to_server,
)
from rich.console import Console
from rich.table import Table

from flyte import logger


def is_terminal_phase(phase: TaskExecution.Phase) -> bool:
    """
    Return true if the phase is terminal.
    """
    return phase in [TaskExecution.SUCCEEDED, TaskExecution.ABORTED, TaskExecution.FAILED]


def convert_to_flyte_phase(state: str) -> TaskExecution.Phase:
    """
    Convert the state from the connector to the phase in flyte.
    """
    state = state.lower()
    if state in ["failed", "timeout", "timedout", "canceled", "cancelled", "skipped"]:
        return TaskExecution.FAILED
    if state in ["internal_error"]:
        return TaskExecution.RETRYABLE_FAILED
    elif state in ["done", "succeeded", "success", "completed"]:
        return TaskExecution.SUCCEEDED
    elif state in ["running", "terminating"]:
        return TaskExecution.RUNNING
    elif state in ["pending"]:
        return TaskExecution.INITIALIZING
    raise ValueError(f"Unrecognized state: {state}")


async def _start_grpc_server(
    port: int, prometheus_port: int, worker: int, timeout: int | None, modules: List[str] | None
):
    try:
        from flyte.connectors._server import (
            AsyncConnectorService,
            ConnectorMetadataService,
        )
    except ImportError as e:
        raise ImportError(
            "Flyte connector dependencies are not installed."
            " Please install it using `pip install flyteplugins-connector`"
        ) from e

    click.secho("🚀 Starting the connector service...")
    _load_connectors(modules)
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

    connectors = ConnectorRegistry._list_connectors()

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


def _load_connectors(modules: List[str] | None):
    plugins = entry_points(group="flyte.connectors")
    for ep in plugins:
        try:
            logger.info(f"Loading connector: {ep.name}")
            ep.load()
        except Exception as e:
            logger.warning(f"Failed to load type transformer {ep.name} with error: {e}")

    if modules:
        for m in modules:
            importlib.import_module(m)
