from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import TYPE_CHECKING

import click

import flyte

if TYPE_CHECKING:
    # Annotation-only: `from __future__ import annotations` keeps this out of the
    # runtime import path, so `flyte-mcp --help` still works without the mcp extra.
    from flyte.ai.mcp._mcp_app import MCPTransport

_FLYTE_SDK_REPO = "https://github.com/flyteorg/flyte-sdk.git"
_UNIONAI_EXAMPLES_REPO = "https://github.com/unionai/unionai-examples.git"
_LLMS_TXT_URL = "https://www.union.ai/docs/v2/union/llms.txt"

# Persistent on-disk cache for the search corpora so subsequent ``flyte-mcp``
# invocations don't re-clone the repos or re-download ``llms.txt``. Pass
# ``--refresh-cache`` (or delete this directory) to force a refresh.
_MCP_CACHE_DIR = pathlib.Path.home() / ".flyte" / "mcp"


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def _csv_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str] | None:
    return _comma_list(value)


def _central_mode_default() -> bool:
    """Whether ``FLYTE_MCP_CENTRAL`` asks for multi-tenant central mode.

    Read lazily (rather than as a click ``default``) so tests and deployments can set the env
    var after this module is imported.
    """
    from flyte._utils import str2bool
    from flyte.ai.mcp._flyte_mcp_app import CENTRAL_MODE_ENV_VAR

    return str2bool(os.environ.get(CENTRAL_MODE_ENV_VAR, ""))


def _host_allowlist_configured(allowed_hosts: list[str] | None) -> bool:
    """Whether a Host/Origin allowlist is available from the flag or the environment.

    Mirrors ``FlyteMCPAppEnvironment.resolved_allowed_hosts`` / ``resolved_allowed_origins``:
    either list turns MCP's DNS-rebinding protection on, so either satisfies ``--central``.
    Read lazily so a deployment can set the env vars after this module is imported.
    """
    from flyte.ai.mcp._flyte_mcp_app import ALLOWED_HOSTS_ENV_VAR, ALLOWED_ORIGINS_ENV_VAR

    if allowed_hosts:
        return True
    return any(_comma_list(os.environ.get(var)) for var in (ALLOWED_HOSTS_ENV_VAR, ALLOWED_ORIGINS_ENV_VAR))


def _shallow_clone(repo_url: str, dest: pathlib.Path) -> None:
    """Shallow-clone ``repo_url`` into ``dest``. Raises ``click.ClickException`` on failure."""
    if shutil.which("git") is None:
        raise click.ClickException(
            "`git` is required to fetch the MCP search corpus. Install git or pass "
            "--sdk-examples-path / --docs-examples-path explicitly to skip cloning."
        )
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Failed to clone {repo_url}: {e.stderr.strip() or e}") from e


def _download(url: str, dest: pathlib.Path) -> None:
    """Download ``url`` to ``dest``. Raises ``click.ClickException`` on failure.

    Sends a browser-like ``User-Agent`` because some origins (e.g. union.ai)
    return ``403`` to the default ``Python-urllib/x.y`` UA.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "flyte-mcp-server/1.0",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as e:
        raise click.ClickException(f"Failed to download {url}: {type(e).__name__}: {e}") from e


def _ensure_cached_clone(repo_url: str, dest: pathlib.Path, *, refresh: bool = False) -> None:
    """Ensure ``dest`` contains a shallow clone of ``repo_url``.

    No-ops when ``dest`` already exists, unless ``refresh`` is true in which
    case the existing cache entry is evicted and re-cloned. The clone is
    staged into a sibling ``<dest>.partial`` directory and atomically renamed
    into place so an interrupted clone (Ctrl+C, network failure) never leaves
    a half-populated cache entry behind.
    """
    if refresh and dest.exists():
        click.echo(f"Refreshing cached {repo_url} at {dest}")
        shutil.rmtree(dest, ignore_errors=True)
    if dest.exists():
        click.echo(f"Using cached {repo_url} at {dest}")
        return
    click.echo(f"Cloning {repo_url} into {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"{dest.name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _shallow_clone(repo_url, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(dest)


def _ensure_cached_download(url: str, dest: pathlib.Path, *, refresh: bool = False) -> None:
    """Ensure ``dest`` contains the bytes of ``url``.

    No-ops when ``dest`` already exists, unless ``refresh`` is true in which
    case the cached file is removed and re-downloaded. Downloads to
    ``<dest>.partial`` first and atomically renames into place so an
    interrupted download never leaves a truncated file in the cache.
    """
    if refresh and dest.exists():
        click.echo(f"Refreshing cached {url} at {dest}")
        dest.unlink(missing_ok=True)
    if dest.exists():
        click.echo(f"Using cached {url} at {dest}")
        return
    click.echo(f"Downloading {url} to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"{dest.name}.partial"
    try:
        _download(url, staging)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    staging.rename(dest)


def _prepare_search_corpus(
    cache_dir: pathlib.Path,
    *,
    fetch_sdk_examples: bool,
    fetch_docs_examples: bool,
    fetch_full_docs: bool,
    refresh: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """Populate (or reuse) the on-disk search corpus cache under ``cache_dir``.

    Mirrors the layout baked into :data:`flyte.ai.mcp._flyte_mcp_app.DEFAULT_IMAGE`
    so a locally-run MCP server has the same content available as the remote
    deployment. Each asset is cloned/downloaded only if it isn't already cached;
    pass ``refresh=True`` to evict the existing entries before fetching.

    :return: ``(sdk_examples_path, docs_examples_path, full_docs_path)`` with ``None``
        for any asset that wasn't requested (because the caller passed a CLI override
        or the corresponding search tool isn't enabled).
    """
    sdk_examples_path: str | None = None
    docs_examples_path: str | None = None
    full_docs_path: str | None = None

    if fetch_sdk_examples:
        sdk_repo = cache_dir / "flyte-sdk"
        _ensure_cached_clone(_FLYTE_SDK_REPO, sdk_repo, refresh=refresh)
        sdk_examples_path = str(sdk_repo / "examples")

    if fetch_docs_examples:
        docs_repo = cache_dir / "unionai-examples"
        _ensure_cached_clone(_UNIONAI_EXAMPLES_REPO, docs_repo, refresh=refresh)
        docs_examples_path = str(docs_repo / "v2")

    if fetch_full_docs:
        llms = cache_dir / "llms.txt"
        _ensure_cached_download(_LLMS_TXT_URL, llms, refresh=refresh)
        full_docs_path = str(llms)

    return sdk_examples_path, docs_examples_path, full_docs_path


@click.command(
    help=(
        "Serve a Flyte MCP server locally, over HTTP (FastMCP + Starlette) or over stdio.\n"
        "\n"
        "Use --transport stdio when an MCP client launches this as a subprocess; the "
        "default streamable-http binds --port instead."
    )
)
@click.option("--name", default="flyte-mcp-server", show_default=True, help="App name.")
@click.option("--title", default=None, help="Optional MCP server title (defaults to --name).")
@click.option("--instructions", default=None, help="Optional MCP server instructions string.")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="streamable-http",
    show_default=True,
    help="MCP transport. 'stdio' speaks JSON-RPC on stdin/stdout and ignores --port/--mcp-mount-path.",
)
@click.option("--port", type=int, default=8080, show_default=True, help="HTTP port to bind (HTTP transports only).")
@click.option("--mcp-mount-path", default="/flyte-mcp", show_default=True, help="Mount path for MCP endpoint.")
@click.option(
    "--tool-groups",
    default=None,
    callback=_csv_callback,
    help="Comma-separated tool groups to enable (mutually exclusive with --tools).",
)
@click.option(
    "--tools",
    default=None,
    callback=_csv_callback,
    help="Comma-separated individual tools to enable (mutually exclusive with --tool-groups).",
)
@click.option(
    "--read-only",
    "read_only",
    is_flag=True,
    default=False,
    help=(
        "Keep only the tools annotated readOnlyHint=True. Applied after --tool-groups/--tools, "
        "so it narrows whatever those selected."
    ),
)
@click.option("--sdk-examples-path", default=None, help="Path for search_flyte_sdk_examples.")
@click.option("--docs-examples-path", default=None, help="Path for search_flyte_docs_examples.")
@click.option("--full-docs-path", default=None, help="Path for search_full_docs.")
@click.option(
    "--refresh-cache",
    is_flag=True,
    default=False,
    help=(
        "Force a re-fetch of the search corpus cache at ~/.flyte/mcp. "
        "Only the assets the CLI is about to fetch are refreshed; entries you "
        "override with --sdk-examples-path / --docs-examples-path / --full-docs-path "
        "are left untouched."
    ),
)
@click.option(
    "--init-from-config/--no-init-from-config",
    default=True,
    show_default=True,
    help="Initialize Flyte config before serving.",
)
@click.option(
    "--requires-auth",
    default=False,
    is_flag=True,
    show_default=True,
    help="Require authentication for the MCP server.",
)
@click.option(
    "--central",
    "central",
    default=None,
    is_flag=True,
    help=(
        "Serve in multi-tenant central mode: each request's Authorization: Bearer <api-key> "
        "selects the Flyte tenant, instead of the process binding one endpoint at startup. "
        "Defaults to the FLYTE_MCP_CENTRAL environment variable."
    ),
)
@click.option(
    "--allowed-endpoint-suffixes",
    default=None,
    callback=_csv_callback,
    help=(
        "Comma-separated endpoint suffixes reachable in central mode; setting this (or "
        "FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES) REPLACES the built-in defaults. By default only "
        "Union-operated control planes are reachable: <org>.hosted.unionai.cloud, "
        "<org>.<region>.unionai.cloud (us-west-2, eu-west-1, eu-west-2, eu-central-1), "
        "<org>.s.union.ai and <org>.us-east-2.s.union.ai, with <org> a single DNS label. "
        "A self-hosted or private deployment lists its own control planes here. Deployed-app "
        "hostnames (<name>.apps.<org>....) are always rejected; they are not control planes."
    ),
)
@click.option(
    "--allowed-hosts",
    default=None,
    callback=_csv_callback,
    help=(
        "Comma-separated Host header allowlist. Setting it (or FLYTE_MCP_ALLOWED_HOSTS) turns on "
        "MCP's DNS-rebinding protection. Required with --central."
    ),
)
def main(
    name: str,
    title: str | None,
    instructions: str | None,
    transport: "MCPTransport",
    port: int,
    mcp_mount_path: str,
    tool_groups: list[str] | None,
    tools: list[str] | None,
    read_only: bool,
    sdk_examples_path: str | None,
    docs_examples_path: str | None,
    full_docs_path: str | None,
    refresh_cache: bool,
    init_from_config: bool,
    requires_auth: bool,
    central: bool | None,
    allowed_endpoint_suffixes: list[str] | None,
    allowed_hosts: list[str] | None,
) -> None:
    central_mode = _central_mode_default() if central is None else central

    stdio = transport == "stdio"

    if stdio and central_mode:
        raise click.ClickException(
            "--central requires an HTTP transport. stdio serves the single process that "
            "launched it, so there is no per-request credential to select a tenant from."
        )

    if central_mode and transport != "streamable-http":
        # SSE keeps one stateful session per GET /sse, and tool calls posted to that session
        # execute in the *session opener's* context — so a second tenant's credential would be
        # authenticated and then ignored, running their calls as the opener. Only the stateless
        # streamable-http transport resolves the tenant per request.
        raise click.ClickException(
            f"--central requires --transport streamable-http (got {transport!r}). Stateful "
            "transports bind tool execution to the session opener's tenant."
        )

    if central_mode and not _host_allowlist_configured(allowed_hosts):
        # A central server is a public multi-tenant endpoint. MCP only enables DNS-rebinding
        # protection once an allowlist exists, so serving without one silently answers requests
        # for any Host. Fail here rather than at import time inside the app environment, so the
        # message names the flag the operator actually forgot.
        from flyte.ai.mcp._flyte_mcp_app import ALLOWED_HOSTS_ENV_VAR

        raise click.ClickException(
            f"--central requires a Host allowlist: pass --allowed-hosts (or set "
            f"{ALLOWED_HOSTS_ENV_VAR}) to the hostname this server is served on, e.g. "
            f"--allowed-hosts mcp.union.ai,mcp.union.ai:443. Without one, MCP's DNS-rebinding "
            f"protection stays off."
        )

    if central_mode and init_from_config:
        # A central server has no single tenant, so there is nothing useful for a config
        # file to bind; skipping it also avoids failing at startup on a machine with no
        # Flyte config at all.
        click.echo("--central: skipping init-from-config (the tenant comes from each request).", err=True)
        init_from_config = False

    if stdio and requires_auth:
        click.echo(
            "warning: --requires-auth has no effect with --transport stdio. The server "
            "speaks to whichever process launched it and runs with your own credentials.",
            err=True,
        )

    # Under stdio, stdout is the JSON-RPC channel: a single stray byte of setup chatter
    # (click.echo here, a print or progress bar in a dependency) makes the client's first
    # parse fail. Divert stdout to stderr while building the environment -- but *not*
    # while running it, since run_stdio() needs the real stdout to answer on.
    with contextlib.redirect_stdout(sys.stderr) if stdio else contextlib.nullcontext():
        env = _build_env(
            name=name,
            title=title,
            instructions=instructions,
            transport=transport,
            port=port,
            mcp_mount_path=mcp_mount_path,
            tool_groups=tool_groups,
            tools=tools,
            read_only=read_only,
            sdk_examples_path=sdk_examples_path,
            docs_examples_path=docs_examples_path,
            full_docs_path=full_docs_path,
            refresh_cache=refresh_cache,
            init_from_config=init_from_config,
            requires_auth=requires_auth,
            central_mode=central_mode,
            allowed_endpoint_suffixes=allowed_endpoint_suffixes,
            allowed_hosts=allowed_hosts,
        )

    if stdio:
        env.run_stdio()
        return

    _serve_http(env, mcp_mount_path)


def _build_env(
    *,
    name: str,
    title: str | None,
    instructions: str | None,
    transport: "MCPTransport",
    port: int,
    mcp_mount_path: str,
    tool_groups: list[str] | None,
    tools: list[str] | None,
    read_only: bool,
    sdk_examples_path: str | None,
    docs_examples_path: str | None,
    full_docs_path: str | None,
    refresh_cache: bool,
    init_from_config: bool,
    requires_auth: bool,
    central_mode: bool = False,
    allowed_endpoint_suffixes: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
):
    """Initialize Flyte, materialize the search corpus, and build the app environment."""
    if init_from_config:
        flyte.init_from_config()

    from flyte.ai.mcp import FlyteMCPAppEnvironment, resolve_tools

    # Figure out which search tools will actually be registered so we only fetch
    # the corpora we'll use. ``resolve_tools`` also validates the CLI flags
    # (unknown groups/tools, mutually-exclusive --tools/--tool-groups) so we get
    # a clear error before any network I/O.
    enabled_tools = resolve_tools(tool_groups, tools, read_only=read_only)

    fetch_sdk_examples = sdk_examples_path is None and "search_flyte_sdk_examples" in enabled_tools
    fetch_docs_examples = docs_examples_path is None and "search_flyte_docs_examples" in enabled_tools
    fetch_full_docs = full_docs_path is None and "search_full_docs" in enabled_tools

    if fetch_sdk_examples or fetch_docs_examples or fetch_full_docs:
        click.echo(f"Using search corpus cache at {_MCP_CACHE_DIR}")
        fetched_sdk, fetched_docs, fetched_llms = _prepare_search_corpus(
            _MCP_CACHE_DIR,
            fetch_sdk_examples=fetch_sdk_examples,
            fetch_docs_examples=fetch_docs_examples,
            fetch_full_docs=fetch_full_docs,
            refresh=refresh_cache,
        )
        sdk_examples_path = sdk_examples_path or fetched_sdk
        docs_examples_path = docs_examples_path or fetched_docs
        full_docs_path = full_docs_path or fetched_llms
    elif refresh_cache:
        click.echo("--refresh-cache had no effect: no cached corpus assets needed for this configuration.")

    return FlyteMCPAppEnvironment(
        name=name,
        title=title,
        instructions=instructions,
        transport=transport,
        port=port,
        mcp_mount_path=mcp_mount_path,
        tool_groups=tool_groups,
        tools=tools,
        read_only=read_only,
        sdk_examples_path=sdk_examples_path,
        docs_examples_path=docs_examples_path,
        full_docs_path=full_docs_path,
        requires_auth=requires_auth,
        central_mode=central_mode,
        allowed_endpoint_suffixes=allowed_endpoint_suffixes,
        allowed_hosts=allowed_hosts,
    )


def _serve_http(env, mcp_mount_path: str) -> None:
    """Serve an HTTP-transport environment locally until interrupted."""
    serve_ctx = flyte.with_servecontext(mode="local")
    app = serve_ctx.serve(env)

    # ``_serve_local`` starts the server in a background thread/subprocess
    # and returns immediately; block the foreground here so the CLI process
    # keeps the server alive until the user interrupts.
    app.activate(wait=True)
    suffix = "/sse" if env.transport == "sse" else "/mcp"
    click.echo(f"MCP server listening at {app.endpoint}{mcp_mount_path}{suffix}")
    click.echo("Press Ctrl+C to stop.")
    try:
        while not app.is_deactivated():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.deactivate(wait=True)
