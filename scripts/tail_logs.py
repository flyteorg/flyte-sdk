"""Test tailing logs from the Flyte Python SDK.

Supports splitting the controlplane (auth, run lookup) and dataplane (logs)
endpoints. When ``--controlplane-endpoint`` is provided, all controlplane calls
go there while ``--endpoint`` is used only for the streaming logs RPC. When
omitted, ``--endpoint`` is used for everything.
"""

import rich_click as click
from flyteidl2.workflow.run_logs_service_connect import RunLogsServiceClient

import flyte
from flyte._initialize import get_client
from flyte.remote import Run


@click.command()
@click.option(
    "--endpoint",
    required=True,
    help="Dataplane endpoint used for the tail-logs RPC (e.g. dns:///...).",
)
@click.option(
    "--controlplane-endpoint",
    default=None,
    help="Optional controlplane endpoint for auth and run lookup. "
    "Defaults to --endpoint if not provided.",
)
@click.option("--project", required=True, help="Project name.")
@click.option("--domain", required=True, help="Domain name.")
@click.option("--run-id", required=True, help="Run ID to tail logs for.")
@click.option("--max-lines", default=50, show_default=True, type=int)
@click.option("--show-ts/--no-show-ts", default=True, show_default=True)
@click.option("--raw/--pretty", default=False, show_default=True)
def main(
    endpoint: str,
    controlplane_endpoint: str | None,
    project: str,
    domain: str,
    run_id: str,
    max_lines: int,
    show_ts: bool,
    raw: bool,
) -> None:
    # Initialize against the controlplane so auth + run lookups target it.
    init_endpoint = controlplane_endpoint or endpoint
    flyte.init(endpoint=init_endpoint, project=project, domain=domain)

    # If a split endpoint was requested, override the logs service to point at
    # the dataplane endpoint while reusing the controlplane's interceptors
    # (auth, retries, metadata) and HTTP client.
    if controlplane_endpoint and controlplane_endpoint != endpoint:
        client = get_client()
        session_cfg = client.session_config
        from flyte.remote._client.auth._session import normalize_rpc_endpoint

        dataplane_address = normalize_rpc_endpoint(endpoint, insecure=session_cfg.insecure)
        client._log_service = RunLogsServiceClient(
            address=dataplane_address,
            interceptors=session_cfg.interceptors,
            http_client=session_cfg.http_client,
        )

    run = Run.get(run_id)
    run.show_logs(max_lines=max_lines, show_ts=show_ts, raw=raw)


if __name__ == "__main__":
    main()
