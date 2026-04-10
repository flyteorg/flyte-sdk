"""Smoke-test ``DataProxyService.CreateUploadLocation`` against a split controlplane/dataplane backend.

Target: ``flyteidl2.dataproxy.DataProxyService.CreateUploadLocation``

Why this one:
  - Unary (simpler than streaming), so any error is from the backend handler
    itself rather than stream teardown / trailers.
  - It's a side-effecting RPC that returns a real signed URL, so a 200
    response proves the service works end-to-end.

Supports splitting the controlplane (auth) and the target endpoint. When
``--controlplane-endpoint`` is provided, all auth calls go there while
``--endpoint`` is used only for the target RPC. When omitted, ``--endpoint`` is
used for everything.
"""

import asyncio
import hashlib

import rich_click as click
from flyteidl2.dataproxy.dataproxy_service_connect import DataProxyServiceClient
from flyteidl2.dataproxy.dataproxy_service_pb2 import CreateUploadLocationRequest
from google.protobuf.duration_pb2 import Duration

import flyte
from flyte._initialize import get_client


async def _create_upload_location(
    client: DataProxyServiceClient, request: CreateUploadLocationRequest
) -> None:
    resp = await client.create_upload_location(request)
    print(f"native_url: {resp.native_url}")
    print(f"signed_url: {resp.signed_url}")


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
@click.option("--project", required=True, help="Project name.")
@click.option("--domain", required=True, help="Domain name.")
@click.option("--org", default="", help="Org name (inferred from controlplane if empty).")
@click.option(
    "--filename",
    default="smoke-test.txt",
    show_default=True,
    help="Filename to request a signed upload URL for.",
)
@click.option(
    "--filename-root",
    default="",
    help="Optional filename prefix/root for the upload location.",
)
@click.option(
    "--expires-in",
    default=3600,
    show_default=True,
    type=int,
    help="Requested URL lifetime in seconds.",
)
@click.option(
    "--content-length",
    default=0,
    show_default=True,
    type=int,
    help="Optional advertised content length.",
)
def main(
    endpoint: str,
    controlplane_endpoint: str | None,
    project: str,
    domain: str,
    org: str,
    filename: str,
    filename_root: str,
    expires_in: int,
    content_length: int,
) -> None:
    init_endpoint = controlplane_endpoint or endpoint
    flyte.init(endpoint=init_endpoint, project=project, domain=domain)

    flyte_client = get_client()
    session_cfg = flyte_client.session_config
    from flyte._utils.org_discovery import org_from_endpoint
    from flyte.remote._client.auth._session import normalize_rpc_endpoint

    effective_org = org or org_from_endpoint(init_endpoint) or ""

    target = endpoint if (controlplane_endpoint and controlplane_endpoint != endpoint) else init_endpoint
    address = normalize_rpc_endpoint(target, insecure=session_cfg.insecure)
    dataproxy_client = DataProxyServiceClient(
        address=address,
        interceptors=session_cfg.interceptors,
        http_client=session_cfg.http_client,
    )

    content_md5 = hashlib.md5(filename.encode("utf-8")).digest()
    request = CreateUploadLocationRequest(
        project=project,
        domain=domain,
        filename=filename,
        expires_in=Duration(seconds=expires_in),
        content_md5=content_md5,
        filename_root=filename_root,
        org=effective_org,
        content_length=content_length,
    )
    asyncio.run(_create_upload_location(dataproxy_client, request))


if __name__ == "__main__":
    main()
