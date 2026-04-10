"""Smoke-test ``SupportService.TailSystemLogs`` against a split controlplane/dataplane backend.

This is a server-streaming RPC that tails system logs for a given application
(e.g. flytepropeller, union-operator) or pod label selector in a specific
cluster and namespace.

The stream first sends a ``LogContainersList`` message listing the containers
being tailed, then streams ``LogLinesBatch`` messages with the actual log
lines.

Supports splitting the controlplane (auth) and the target endpoint. When
``--controlplane-endpoint`` is provided, all auth calls go there while
``--endpoint`` is used only for the target RPC. When omitted, ``--endpoint`` is
used for everything.
"""

import asyncio
import datetime
import sys

import rich_click as click
from google.protobuf.timestamp_pb2 import Timestamp

# The ``support`` proto-python package is generated in the cloud repo but not
# published as a pip-installable package. Add the generation root to sys.path
# so we can import it directly.
_CLOUD_GEN_PB_PYTHON = str(
    __import__("pathlib").Path.home()
    / "Workspace/repos/unionai/cloud/gen/pb_python"
)
if _CLOUD_GEN_PB_PYTHON not in sys.path:
    sys.path.insert(0, _CLOUD_GEN_PB_PYTHON)

from support.payload_pb2 import Application, TailSystemLogsRequest  # noqa: E402
from support.service_connect import SupportServiceClient  # noqa: E402

import flyte  # noqa: E402
from flyte._initialize import get_client  # noqa: E402

# Map CLI-friendly names to the proto enum values.
_APPLICATION_NAMES = {
    "unspecified": Application.APPLICATION_UNSPECIFIED,
    "union-operator": Application.APPLICATION_UNION_OPERATOR,
    "flyte-propeller": Application.APPLICATION_FLYTE_PROPELLER,
}


async def _tail_system_logs(
    client: SupportServiceClient,
    request: TailSystemLogsRequest,
    max_messages: int,
) -> None:
    message_count = 0
    async for resp in client.tail_system_logs(request):
        message_count += 1

        if resp.HasField("containers"):
            containers = resp.containers.containers
            print(f"\n--- containers ({len(containers)}) ---")
            for i, c in enumerate(containers):
                print(
                    f"  [{i}] pod={c.kubernetes_pod_name} "
                    f"container={c.kubernetes_container_name} "
                    f"namespace={c.kubernetes_namespace}"
                )

        elif resp.HasField("log_lines_batch"):
            batch = resp.log_lines_batch
            for log_lines in batch.logs:
                container_idx = log_lines.container_index
                container_label = ""
                if log_lines.HasField("container"):
                    c = log_lines.container
                    container_label = f" [{c.kubernetes_pod_name}/{c.kubernetes_container_name}]"

                # The server may populate either `lines` (plain) or
                # `structured_lines` (with richer metadata). Handle both.
                lines = log_lines.lines or log_lines.structured_lines
                for line in lines:
                    ts = ""
                    if line.HasField("timestamp"):
                        ts = f"{line.timestamp.ToDatetime().isoformat()} "
                    print(f"{ts}container_idx={container_idx}{container_label}: {line.message}")

        if 0 < max_messages <= message_count:
            break

    print(f"\n--- stream ended after {message_count} message(s) ---")


@click.command()
@click.option(
    "--endpoint",
    required=True,
    help="Dataplane (or target) endpoint used for the RPC (e.g. dns:///...).",
)
@click.option(
    "--controlplane-endpoint",
    default=None,
    help="Optional controlplane endpoint for auth. Defaults to --endpoint.",
)
@click.option("--cluster-name", required=True, help="Cluster name to tail logs from.")
@click.option("--namespace", required=True, help="Kubernetes namespace to tail logs from.")
@click.option(
    "--application",
    type=click.Choice(list(_APPLICATION_NAMES.keys()), case_sensitive=False),
    default=None,
    help="Application to tail logs for (mutually exclusive with --pod-label-selector).",
)
@click.option(
    "--pod-label-selector",
    default=None,
    help="Pod label selector to filter logs (mutually exclusive with --application).",
)
@click.option(
    "--node-name",
    default="",
    help="Optional Kubernetes node name to filter pods by.",
)
@click.option(
    "--start-time-minutes-ago",
    default=5,
    show_default=True,
    type=int,
    help="How many minutes in the past to start tailing from.",
)
@click.option(
    "--max-messages",
    default=0,
    show_default=True,
    type=int,
    help="Stop after receiving this many stream messages (0 = unlimited).",
)
def main(
    endpoint: str,
    controlplane_endpoint: str | None,
    cluster_name: str,
    namespace: str,
    application: str | None,
    pod_label_selector: str | None,
    node_name: str,
    start_time_minutes_ago: int,
    max_messages: int,
) -> None:
    if application and pod_label_selector:
        raise click.UsageError("--application and --pod-label-selector are mutually exclusive.")
    if not application and not pod_label_selector:
        raise click.UsageError("One of --application or --pod-label-selector is required.")

    init_endpoint = controlplane_endpoint or endpoint
    flyte.init(endpoint=init_endpoint)

    flyte_client = get_client()
    session_cfg = flyte_client.session_config
    from flyte.remote._client.auth._session import normalize_rpc_endpoint

    target = endpoint if (controlplane_endpoint and controlplane_endpoint != endpoint) else init_endpoint
    address = normalize_rpc_endpoint(target, insecure=session_cfg.insecure)
    support_client = SupportServiceClient(
        address=address,
        interceptors=session_cfg.interceptors,
        http_client=session_cfg.http_client,
    )

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start = now - datetime.timedelta(minutes=start_time_minutes_ago)
    start_ts = Timestamp()
    start_ts.FromDatetime(start)

    kwargs: dict = {
        "cluster_name": cluster_name,
        "namespace": namespace,
        "start_time": start_ts,
    }
    if node_name:
        kwargs["node_name"] = node_name
    if application:
        kwargs["application"] = _APPLICATION_NAMES[application.lower()]
    else:
        kwargs["pod_label_selector"] = pod_label_selector

    request = TailSystemLogsRequest(**kwargs)
    asyncio.run(_tail_system_logs(support_client, request, max_messages))


if __name__ == "__main__":
    main()
