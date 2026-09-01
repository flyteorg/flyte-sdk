import asyncio
import errno
import json
import sys

import rich_click as click

import flyte.remote as remote

from . import _common as common

# Headers that must not be forwarded verbatim across a proxy hop.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@click.group(name="proxy")
def proxy():
    """Proxy a local port into a Flyte App through the authenticated app edge."""


@proxy.command(cls=common.CommandBase)
@click.argument("name", type=str, required=False)
@click.option("--url", "url", type=str, default=None, help="Proxy this app URL directly; skip name resolution.")
@click.option("--port", type=int, default=8600, help="Local port to listen on (0 = pick a free port).")
@click.option("--address", type=str, default="127.0.0.1", help="Local bind address; non-loopback triggers a warning.")
@click.option(
    "--emit-mcp-config",
    is_flag=True,
    default=False,
    help="Print a generic HTTP-MCP config block for the local endpoint.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Log each proxied request (never the token).")
@click.pass_obj
def app(
    cfg: common.CLIConfig,
    name: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    url: str | None = None,
    port: int = 8600,
    address: str = "127.0.0.1",
    emit_mcp_config: bool = False,
    verbose: bool = False,
):
    """
    Open an authenticated localhost proxy into a no-auth Flyte App.

    Reuses the same Union auth (with auto-refresh) the CLI uses, injecting a fresh bearer on every
    request, so a local HTTP client — a Grafana/Prometheus MCP, curl, a browser — reaches an
    edge-gated app with no token handling. Think: kubectl port-forward for Flyte Apps.

    Foreground; Ctrl-C to stop.
    """
    cfg.init(project=project, domain=domain)

    if url:
        target = url
    elif name:
        target = remote.App.get(name=name).endpoint
    else:
        raise click.UsageError("Provide an app NAME or --url.")
    target = target.rstrip("/")

    label = name or target
    try:
        asyncio.run(_serve(cfg, target, label, address, port, emit_mcp_config, verbose))
    except KeyboardInterrupt:
        pass


def _build_authenticator(cfg: common.CLIConfig):
    import typing

    from flyte.remote._client.auth._authenticators.factory import get_async_authenticator
    from flyte.remote._client.auth._client_config import AuthType, RemoteClientConfigStore
    from flyte.remote._client.auth._session import normalize_rpc_endpoint

    plat = cfg.config.platform
    if not plat.endpoint:
        raise click.UsageError("No endpoint configured; set one via config or FLYTECTL_CONFIG.")
    insecure = getattr(plat, "insecure", False)
    # Config endpoint is gRPC-style (bare host); the OIDC-metadata client needs an http(s) base.
    endpoint = normalize_rpc_endpoint(plat.endpoint, insecure=insecure)
    auth_type = typing.cast(AuthType, getattr(plat, "auth_mode", None) or "Pkce")
    return get_async_authenticator(
        endpoint=endpoint,
        cfg_store=RemoteClientConfigStore(endpoint),
        auth_type=auth_type,
        insecure_skip_verify=getattr(plat, "insecure_skip_verify", False),
        ca_cert_file_path=getattr(plat, "ca_cert_file_path", None),
    )


def _filter_request_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP and k.lower() != "authorization"}


def _filter_response_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def _serve(cfg, target, label, address, port, emit_mcp_config, verbose):
    from aiohttp import ClientSession, ClientTimeout, web

    authenticator = _build_authenticator(cfg)
    # Prime credentials (load from keyring / run the auth flow) before we start serving.
    if authenticator.get_credentials() is None:
        await authenticator.refresh_credentials()

    # auto_decompress=False -> stream upstream bytes verbatim (Content-Encoding preserved).
    # total=None -> allow long-lived SSE streams (MCP streamable-HTTP, Grafana Live).
    upstream = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)

    async def handle(request: "web.Request"):
        body = await request.read()
        fwd = _filter_request_headers(request.headers)
        up_url = target + request.raw_path

        async def send(refresh: bool):
            if refresh:
                await authenticator.refresh_credentials()
            ah = await authenticator.get_auth_headers()
            hdrs = dict(fwd)
            if ah:
                # Normalize the injected bearer onto the standard `authorization` header.
                # The SDK auth flow emits its bearer under the CP's configured metadata key
                # (typically the gRPC-style `flyte-authorization`), but a Flyte App's HTTP
                # auth edge (union-apps) reads only `authorization` and never sees
                # `flyte-authorization`, so the request is rejected. Scope this to the
                # authenticator's own headers — never rewrite forwarded client headers.
                for k, v in ah.headers.items():
                    hdrs.pop(k, None)
                    if isinstance(v, str) and v.startswith("Bearer "):
                        hdrs["authorization"] = v
                    else:
                        hdrs[k] = v
            return await upstream.request(
                request.method, up_url, headers=hdrs, data=body or None, allow_redirects=False
            )

        resp = await send(refresh=False)
        # Stale token: the edge 401s or bounces to /login. Refresh once and retry.
        bounced = resp.status in (302, 307) and "/login" in resp.headers.get("Location", "")
        if resp.status in (401, 403) or bounced:
            resp.release()
            resp = await send(refresh=True)

        out = web.StreamResponse(status=resp.status, headers=_filter_response_headers(resp.headers))
        await out.prepare(request)
        async for chunk in resp.content.iter_any():
            await out.write(chunk)
        await out.write_eof()
        resp.release()
        if verbose:
            click.echo(f"{request.method} {request.path} -> {resp.status}", err=True)
        return out

    server = web.Application()
    server.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(server)
    await runner.setup()
    # Bind loopback on BOTH IPv4 and IPv6 so the proxy owns `localhost` fully.
    # macOS resolves `localhost` to ::1 (IPv6) first, and a wildcard listener in
    # another process (e.g. OrbStack/Docker on *:PORT) otherwise shadows an
    # IPv4-only 127.0.0.1 bind: bind() still succeeds, so the proxy looks healthy
    # yet never receives the request (it lands on the other listener, which
    # resets it). Binding ::1 too turns that silent half-bind into a loud
    # EADDRINUSE. A non-loopback --address is bound as-is (single family).
    loopback = address in ("127.0.0.1", "localhost", "::1", "loopback")
    hosts = ["127.0.0.1", "::1"] if loopback else [address]

    actual = port
    bound: list[str] = []
    for host in hosts:
        try:
            await web.TCPSite(runner, host, actual).start()
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                await runner.cleanup()
                raise click.ClickException(
                    f"Port {actual} is already in use (binding {host} failed). Another "
                    f"process — often a wildcard binder like OrbStack or Docker — holds it "
                    f"and would silently shadow this proxy. Re-run with --port <free-port>."
                )
            if host == "::1" and e.errno in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT):
                click.secho(
                    f"note: IPv6 loopback unavailable, binding IPv4 only ({e.strerror}).",
                    fg="yellow",
                    err=True,
                )
                continue
            await runner.cleanup()
            raise
        bound.append(host)
        if port == 0 and actual == 0:
            # First site picked a free port; pin the other family to the same one.
            actual = runner.addresses[0][1]
    if not bound:
        await runner.cleanup()
        raise click.ClickException("Failed to bind any loopback address.")

    display_host = "127.0.0.1" if loopback else address
    local = f"http://{display_host}:{actual}"

    if not loopback:
        click.secho(
            f"WARNING: binding {address} exposes your Union identity to anything that can reach it; prefer 127.0.0.1.",
            fg="yellow",
            err=True,
        )
    identity = _identity(authenticator)
    click.echo(f"Proxying {target}  ->  {local}", err=True)
    click.echo(f"  authenticating as: {identity}", err=True)
    click.echo("  Ctrl-C to stop.", err=True)
    if emit_mcp_config:
        _emit_mcp_config(label, local)

    try:
        await asyncio.Event().wait()  # serve until interrupted (Ctrl-C)
    finally:
        await upstream.close()
        await runner.cleanup()


def _identity(authenticator) -> str:
    creds = authenticator.get_credentials()
    if not creds or not creds.access_token:
        return "<unknown>"
    # Best-effort: decode the JWT payload's sub/email without verifying (display only).
    try:
        import base64

        payload = creds.access_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email") or claims.get("sub") or "<token>"
    except Exception:
        return "<token>"


def _emit_mcp_config(name: str, local: str):
    block = {"mcpServers": {name: {"type": "http", "url": local}}}
    click.echo("", err=True)
    click.echo("# MCP client config (generic HTTP transport) — point your client at the local proxy:", err=True)
    sys.stdout.write(json.dumps(block, indent=2) + "\n")
    sys.stdout.flush()
