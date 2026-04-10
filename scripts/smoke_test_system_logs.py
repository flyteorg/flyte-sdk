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
import logging
import sys

import rich_click as click
from connectrpc import _protocol as _connect_protocol
from connectrpc.errors import ConnectError
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

_diag_log = logging.getLogger("smoke.diag")

# Map CLI-friendly names to the proto enum values.
_APPLICATION_NAMES = {
    "unspecified": Application.APPLICATION_UNSPECIFIED,
    "union-operator": Application.APPLICATION_UNION_OPERATOR,
    "flyte-propeller": Application.APPLICATION_FLYTE_PROPELLER,
}


def _configure_verbose_logging() -> None:
    """Enable DEBUG logging for flyte + the transport stack."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    for name in ("flyte", "connectrpc", "httpx", "httpcore", "smoke.diag"):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _patch_connect_wire_error() -> None:
    """Log the raw HTTP response before Connect swallows it into a terse message."""
    original = _connect_protocol.ConnectWireError.from_response

    def _logging_from_response(response):  # type: ignore[no-untyped-def]
        try:
            body = bytes(response.content) if response.content is not None else b""
        except Exception as exc:
            body = b""
            _diag_log.debug("failed to read response body: %r", exc)
        try:
            body_preview = body[:2048].decode("utf-8", errors="replace")
        except Exception:
            body_preview = repr(body[:2048])
        try:
            headers_dump = dict(response.headers.items())
        except Exception:
            headers_dump = {"<error>": "could not enumerate headers"}
        _diag_log.error(
            "connect wire error: status=%s body_len=%d headers=%s body=%r",
            response.status,
            len(body),
            headers_dump,
            body_preview,
        )
        return original(response)

    _connect_protocol.ConnectWireError.from_response = staticmethod(_logging_from_response)


_REDACTED_HEADER_KEYS = frozenset(
    {
        "authorization",
        "flyte-authorization",
        "cookie",
        "set-cookie",
        "proxy-authorization",
    }
)


def _redact_headers(headers):  # type: ignore[no-untyped-def]
    redacted = {}
    try:
        items = list(headers.allitems())
    except Exception:
        try:
            items = list(headers.items())
        except Exception:
            return {"<error>": "could not enumerate headers"}
    for key, value in items:
        if key.lower() in _REDACTED_HEADER_KEYS:
            redacted[key] = f"<redacted {len(value)} chars>"
        else:
            redacted[key] = value
    return redacted


class RewriteAuthHeaderInterceptor:
    """HACK: Rewrite ``flyte-authorization`` to ``authorization`` on outgoing requests.

    The dogfood dataplane sits behind Cloudflare, which enforces a standard
    ``Authorization: Bearer ...`` header and 401's anything that doesn't have
    one. After a fresh PKCE login, the SDK's auth interceptor sets
    ``flyte-authorization`` instead of ``authorization``, so the edge proxy
    bounces the request before it ever reaches the backend.

    This interceptor runs *after* the auth interceptor in the chain (because
    interceptors are applied in order and we append it after them) and copies
    the bearer token over to ``authorization`` so Cloudflare is happy.
    """

    async def intercept_unary(self, call_next, request, ctx):
        self._rewrite(ctx)
        return await call_next(request, ctx)

    async def intercept_server_stream(self, call_next, request, ctx):
        self._rewrite(ctx)
        async for response in call_next(request, ctx):
            yield response

    async def intercept_client_stream(self, call_next, request, ctx):
        self._rewrite(ctx)
        return await call_next(request, ctx)

    async def intercept_bidi_stream(self, call_next, request, ctx):
        self._rewrite(ctx)
        async for response in call_next(request, ctx):
            yield response

    @staticmethod
    def _rewrite(ctx) -> None:
        headers = ctx.request_headers()
        token = headers.get("flyte-authorization")
        if token is None:
            return
        existing = headers.get("authorization")
        if existing == token:
            return
        headers["authorization"] = token
        _diag_log.debug(
            "rewrote flyte-authorization -> authorization (%d chars)", len(token)
        )


class DiagnosticStreamInterceptor:
    """ConnectRPC interceptor that logs URL, method, headers, and errors for server-streaming RPCs."""

    async def intercept_unary(self, call_next, request, ctx):
        return await call_next(request, ctx)

    async def intercept_server_stream(self, call_next, request, ctx):
        method = ctx.method()
        url = f"/{method.service_name}/{method.name}"
        headers = _redact_headers(ctx.request_headers())
        _diag_log.info(
            "server_stream rpc begin: service=%s method=%s server_address=%s headers=%s",
            method.service_name,
            method.name,
            ctx.server_address(),
            headers,
        )
        try:
            async for response in call_next(request, ctx):
                _diag_log.debug("server_stream rpc message: url=%s", url)
                yield response
            _diag_log.info("server_stream rpc completed: url=%s", url)
        except ConnectError as e:
            try:
                details = [repr(d) for d in e.details]
            except Exception:
                details = ["<error rendering details>"]
            _diag_log.error(
                "server_stream rpc ConnectError: url=%s code=%s message=%r details=%s",
                url,
                e.code,
                e.message,
                details,
            )
            raise
        except Exception as e:
            _diag_log.exception(
                "server_stream rpc unexpected error: url=%s type=%s: %s",
                url,
                type(e).__name__,
                e,
            )
            raise

    async def intercept_client_stream(self, call_next, request, ctx):
        return await call_next(request, ctx)

    async def intercept_bidi_stream(self, call_next, request, ctx):
        async for response in call_next(request, ctx):
            yield response


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
            _diag_log.info("reached max_messages=%d, stopping", max_messages)
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
@click.option(
    "--verbose/--quiet",
    default=True,
    show_default=True,
    help="Enable DEBUG logging and dump raw ConnectRPC transport errors.",
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
    verbose: bool,
) -> None:
    if application and pod_label_selector:
        raise click.UsageError("--application and --pod-label-selector are mutually exclusive.")
    if not application and not pod_label_selector:
        raise click.UsageError("One of --application or --pod-label-selector is required.")

    if verbose:
        _configure_verbose_logging()
        _patch_connect_wire_error()

    init_endpoint = controlplane_endpoint or endpoint
    _diag_log.info(
        "init: controlplane=%s target=%s cluster=%s namespace=%s",
        init_endpoint,
        endpoint,
        cluster_name,
        namespace,
    )
    flyte.init(endpoint=init_endpoint)

    flyte_client = get_client()
    session_cfg = flyte_client.session_config
    from flyte.remote._client.auth._session import normalize_rpc_endpoint

    target = endpoint if (controlplane_endpoint and controlplane_endpoint != endpoint) else init_endpoint
    address = normalize_rpc_endpoint(target, insecure=session_cfg.insecure)
    interceptors = tuple(session_cfg.interceptors) + (RewriteAuthHeaderInterceptor(),)
    if verbose:
        interceptors = interceptors + (DiagnosticStreamInterceptor(),)
    _diag_log.info(
        "SupportServiceClient: address=%s insecure=%s interceptors=%s",
        address,
        session_cfg.insecure,
        [type(i).__name__ for i in interceptors],
    )
    support_client = SupportServiceClient(
        address=address,
        interceptors=interceptors,
        http_client=session_cfg.http_client,
    )

    # Build the request.
    import datetime

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
    _diag_log.info(
        "sending TailSystemLogsRequest: %s", str(request).replace("\n", " ")
    )
    asyncio.run(_tail_system_logs(support_client, request, max_messages))


if __name__ == "__main__":
    main()
