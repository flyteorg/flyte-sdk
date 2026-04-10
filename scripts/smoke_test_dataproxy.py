"""Smoke-test a flyteidl2 RPC against a split controlplane/dataplane backend.

We've cycled through endpoints looking for one the dogfood backend actually
serves end-to-end. Current target:
``flyteidl2.dataproxy.DataProxyService.CreateUploadLocation``.

Why this one:
  - Unary (simpler than streaming), so any error is from the backend handler
    itself rather than stream teardown / trailers.
  - Service #2 in the dogfood registration list.
  - It's a side-effecting RPC that returns a real signed URL, so a 200
    response proves the service works end-to-end (not just that the request
    made it to an error handler).
  - The other dataproxy method, ``UploadInputs``, needs a real run/task
    identifier, which the backend we're testing doesn't appear to have.

Supports splitting the controlplane (auth) and the target endpoint. When
``--controlplane-endpoint`` is provided, all auth calls go there while
``--endpoint`` is used only for the target RPC. When omitted, ``--endpoint`` is
used for everything.
"""

import asyncio
import hashlib
import logging
import sys

import rich_click as click
from connectrpc import _protocol as _connect_protocol
from connectrpc.errors import ConnectError
from flyteidl2.dataproxy.dataproxy_service_connect import DataProxyServiceClient
from flyteidl2.dataproxy.dataproxy_service_pb2 import CreateUploadLocationRequest
from google.protobuf.duration_pb2 import Duration

import flyte
from flyte._initialize import get_client

_diag_log = logging.getLogger("smoke.diag")


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
    """Log the raw HTTP response before Connect swallows it into a terse message.

    ``ConnectWireError.from_response`` only keeps the (possibly generic) HTTP
    status phrase when the body is not a Connect JSON error. Proxies/gateways
    (GCLB, Envoy, Ingress) often return plain ``503 Service Unavailable`` with
    useful info in headers/body (``server``, ``via``, ``x-envoy-*``, an HTML
    error page identifying the proxy). Patch it to dump everything first.
    """
    original = _connect_protocol.ConnectWireError.from_response

    def _logging_from_response(response):  # type: ignore[no-untyped-def]
        try:
            body = bytes(response.content) if response.content is not None else b""
        except Exception as exc:  # pragma: no cover - diagnostic path
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


class DiagnosticUnaryInterceptor:
    """ConnectRPC unary interceptor that logs URL, method, headers, and errors."""

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method()
        url = f"/{method.service_name}/{method.name}"
        headers = _redact_headers(ctx.request_headers())
        _diag_log.info(
            "unary rpc begin: service=%s method=%s server_address=%s headers=%s",
            method.service_name,
            method.name,
            ctx.server_address(),
            headers,
        )
        try:
            response = await call_next(request, ctx)
            _diag_log.info("unary rpc completed: url=%s", url)
            return response
        except ConnectError as e:
            try:
                details = [repr(d) for d in e.details]
            except Exception:
                details = ["<error rendering details>"]
            _diag_log.error(
                "unary rpc ConnectError: url=%s code=%s message=%r details=%s",
                url,
                e.code,
                e.message,
                details,
            )
            raise
        except Exception as e:
            _diag_log.exception(
                "unary rpc unexpected error: url=%s type=%s: %s",
                url,
                type(e).__name__,
                e,
            )
            raise


async def _create_upload_location(
    client: DataProxyServiceClient, request: CreateUploadLocationRequest
) -> None:
    resp = await client.create_upload_location(request)
    _diag_log.info(
        "create_upload_location ok: signed_url_len=%d native_url=%s expires_at=%s headers=%d",
        len(resp.signed_url),
        resp.native_url,
        resp.expires_at.ToDatetime().isoformat() if resp.HasField("expires_at") else "<unset>",
        len(resp.headers),
    )
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
@click.option(
    "--verbose/--quiet",
    default=True,
    show_default=True,
    help="Enable DEBUG logging and dump raw ConnectRPC transport errors.",
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
    verbose: bool,
) -> None:
    if verbose:
        _configure_verbose_logging()
        _patch_connect_wire_error()

    init_endpoint = controlplane_endpoint or endpoint
    _diag_log.info(
        "init: controlplane=%s target=%s org=%s project=%s domain=%s filename=%s",
        init_endpoint,
        endpoint,
        org or "<empty>",
        project,
        domain,
        filename,
    )
    flyte.init(endpoint=init_endpoint, project=project, domain=domain)

    flyte_client = get_client()
    session_cfg = flyte_client.session_config
    from flyte._utils.org_discovery import org_from_endpoint
    from flyte.remote._client.auth._session import normalize_rpc_endpoint

    # If the caller didn't pass --org, fall back to the same heuristic the SDK
    # itself uses so the GetImage handler's org-scoped authz check matches the
    # org the token is actually scoped to.
    effective_org = org or org_from_endpoint(init_endpoint) or ""
    if effective_org != org:
        _diag_log.info(
            "org: caller=%r inferred_from_controlplane=%r using=%r",
            org,
            effective_org,
            effective_org,
        )

    target = endpoint if (controlplane_endpoint and controlplane_endpoint != endpoint) else init_endpoint
    address = normalize_rpc_endpoint(target, insecure=session_cfg.insecure)
    # Append our hack interceptor *after* the SDK auth interceptors so it can
    # observe (and rewrite) the auth header they just injected. The diagnostic
    # interceptor goes last so its log line shows the final, rewritten headers.
    interceptors = tuple(session_cfg.interceptors) + (RewriteAuthHeaderInterceptor(),)
    if verbose:
        interceptors = interceptors + (DiagnosticUnaryInterceptor(),)
    _diag_log.info(
        "DataProxyServiceClient: address=%s insecure=%s interceptors=%s",
        address,
        session_cfg.insecure,
        [type(i).__name__ for i in interceptors],
    )
    dataproxy_client = DataProxyServiceClient(
        address=address,
        interceptors=interceptors,
        http_client=session_cfg.http_client,
    )

    # The dogfood handler's proto-validate requires content_md5 to be exactly
    # 16 bytes (the raw MD5 digest, not a hex string). For a smoke test we
    # don't actually care about the contents, so hash the filename.
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
    _diag_log.info(
        "sending CreateUploadLocationRequest: %s", str(request).replace("\n", " ")
    )
    asyncio.run(_create_upload_location(dataproxy_client, request))


if __name__ == "__main__":
    main()
