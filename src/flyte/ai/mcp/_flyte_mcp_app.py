from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Literal, cast, get_args
from urllib.parse import unquote, urlsplit

import rich.repr
from connectrpc.errors import ConnectError

import flyte
from flyte.ai.mcp._mcp_app import MCPAppEnvironment
from flyte.ai.mcp._protected_resource import (
    WELL_KNOWN_PATH,
    BearerChallengeMiddleware,
    ProtectedResourceMetadata,
    _AuthorizationServerDiscoveryError,
    _require_authority,
    _require_issuer_url,
    _require_resource_path,
    _require_resource_url,
    build_metadata_path,
    discover_authorization_server,
)
from flyte.ai.mcp._tools import (
    LOG_COLLECT_TIMEOUT_S,
    MAX_LOG_LINES,
    ToolError,
    _is_app_allowed,
    _is_task_allowed,
    _is_trigger_allowed,
    _search_files,
    build_tool_functions,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute

# The trailing names are re-exported from ``_tools``: they moved out with the tool bodies, but
# this module stays their public import site.
__all__ = [
    "ALL_MCP_TOOLS",
    "ALL_MCP_TOOL_GROUPS",
    "LOG_COLLECT_TIMEOUT_S",
    "MAX_LOG_LINES",
    "READ_ONLY_MCP_TOOLS",
    "TOOL_GROUP_MAPPING",
    "TOOL_REGISTRY",
    "FlyteMCPAppEnvironment",
    "MCPTool",
    "MCPToolGroup",
    "ProtectedResourceMetadata",
    "ToolError",
    "ToolInfo",
    "_is_app_allowed",
    "_is_task_allowed",
    "_is_trigger_allowed",
    "_search_files",
    "resolve_tools",
]

logger = logging.getLogger(__name__)

#: Comma-separated ``Host`` header allowlist for MCP transport security.
ALLOWED_HOSTS_ENV_VAR = "FLYTE_MCP_ALLOWED_HOSTS"

#: Comma-separated ``Origin`` header allowlist for MCP transport security.
ALLOWED_ORIGINS_ENV_VAR = "FLYTE_MCP_ALLOWED_ORIGINS"

#: Server-wide default project/domain for tools that do not get them per call.
PROJECT_ENV_VAR = "FLYTE_MCP_PROJECT"
DOMAIN_ENV_VAR = "FLYTE_MCP_DOMAIN"


def _env_csv(var: str) -> list[str] | None:
    """Read a comma-separated env var into a list, or `None` when unset/empty."""
    raw = os.environ.get(var)
    if not raw:
        return None
    items = [v.strip() for v in raw.split(",")]
    items = [v for v in items if v]
    return items or None


# ------------------------------
# Tool types & registration table
# ------------------------------

MCPTool = Literal[
    # task
    "run_task",
    "get_task",
    "list_tasks",
    # run
    "get_run",
    "get_run_io",
    "abort_run",
    "list_runs",
    "wait_for_run",
    "rerun_run",
    # action
    "list_actions",
    "get_action",
    "abort_action",
    # logs
    "get_logs",
    # app
    "get_app",
    "list_apps",
    "activate_app",
    "deactivate_app",
    # trigger
    "list_triggers",
    "get_trigger",
    "activate_trigger",
    "deactivate_trigger",
    # project
    "list_projects",
    "get_project",
    # secret
    "list_secrets",
    "create_secret",
    "delete_secret",
    # condition
    "list_conditions",
    "signal_condition",
    # identity
    "whoami",
    # search
    "search_flyte_sdk_examples",
    "search_flyte_docs_examples",
    "search_full_docs",
]

MCPToolGroup = Literal[
    "all",
    "core",
    "task",
    "run",
    "action",
    "logs",
    "app",
    "trigger",
    "project",
    "secret",
    "condition",
    "identity",
    "search",
]

ALL_MCP_TOOL_GROUPS: tuple[MCPToolGroup, ...] = get_args(MCPToolGroup)


@dataclass(frozen=True)
class ToolInfo:
    """Static metadata for one MCP tool: which group it belongs to and how it behaves.

    This is the single source of truth behind `TOOL_GROUP_MAPPING`, the `read_only`
    derivation, and the `mcp.types.ToolAnnotations` attached at registration time —
    so a new tool cannot be added to the server without also declaring its group and hints.
    """

    group: MCPToolGroup
    title: str
    #: The tool only reads; it never mutates the control plane.
    read_only: bool = False
    #: The tool may remove or stop something a caller depends on.
    destructive: bool = False
    #: Calling it twice with the same arguments has the same effect as calling it once.
    idempotent: bool = False

    def annotations(self) -> ToolAnnotations:
        """Build the MCP annotations for this tool.

        `openWorldHint` is always False: every tool talks to the caller's own Flyte control
        plane (or a corpus baked into the image), never to an open-ended set of external hosts.
        """
        from mcp.types import ToolAnnotations

        return ToolAnnotations(
            title=self.title,
            readOnlyHint=self.read_only,
            destructiveHint=self.destructive,
            idempotentHint=self.idempotent,
            openWorldHint=False,
        )


#: name -> (group, title, behavior hints). Registration walks this table.
TOOL_REGISTRY: dict[MCPTool, ToolInfo] = {
    # ---- task ----
    "run_task": ToolInfo("task", "Run a task", read_only=False),
    "get_task": ToolInfo("task", "Get task details", read_only=True, idempotent=True),
    "list_tasks": ToolInfo("task", "List tasks", read_only=True, idempotent=True),
    # ---- run ----
    "get_run": ToolInfo("run", "Get a run", read_only=True, idempotent=True),
    "get_run_io": ToolInfo("run", "Get run inputs and outputs", read_only=True, idempotent=True),
    "abort_run": ToolInfo("run", "Abort a run", destructive=True, idempotent=True),
    "list_runs": ToolInfo("run", "List runs", read_only=True, idempotent=True),
    "wait_for_run": ToolInfo("run", "Wait for a run to finish", read_only=True, idempotent=True),
    "rerun_run": ToolInfo("run", "Re-run a prior run", read_only=False),
    # ---- action ----
    "list_actions": ToolInfo("action", "List the actions of a run", read_only=True, idempotent=True),
    "get_action": ToolInfo("action", "Get action details and timing", read_only=True, idempotent=True),
    "abort_action": ToolInfo("action", "Abort a single action", destructive=True, idempotent=True),
    # ---- logs ----
    "get_logs": ToolInfo("logs", "Read action logs", read_only=True, idempotent=True),
    # ---- app ----
    "get_app": ToolInfo("app", "Get an app", read_only=True, idempotent=True),
    "list_apps": ToolInfo("app", "List apps", read_only=True, idempotent=True),
    "activate_app": ToolInfo("app", "Activate an app", idempotent=True),
    "deactivate_app": ToolInfo("app", "Deactivate an app", destructive=True, idempotent=True),
    # ---- trigger ----
    "list_triggers": ToolInfo("trigger", "List triggers", read_only=True, idempotent=True),
    "get_trigger": ToolInfo("trigger", "Get a trigger", read_only=True, idempotent=True),
    "activate_trigger": ToolInfo("trigger", "Activate a trigger", idempotent=True),
    "deactivate_trigger": ToolInfo("trigger", "Deactivate a trigger", destructive=True, idempotent=True),
    # ---- project ----
    "list_projects": ToolInfo("project", "List projects", read_only=True, idempotent=True),
    "get_project": ToolInfo("project", "Get a project", read_only=True, idempotent=True),
    # ---- secret ----
    "list_secrets": ToolInfo("secret", "List secret names", read_only=True, idempotent=True),
    "create_secret": ToolInfo("secret", "Create a secret", read_only=False),
    "delete_secret": ToolInfo("secret", "Delete a secret", destructive=True, idempotent=True),
    # ---- condition ----
    "list_conditions": ToolInfo("condition", "List the conditions of a run", read_only=True, idempotent=True),
    "signal_condition": ToolInfo("condition", "Signal a waiting condition", read_only=False),
    # ---- identity ----
    "whoami": ToolInfo("identity", "Show the caller's identity", read_only=True, idempotent=True),
    # ---- search ----
    "search_flyte_sdk_examples": ToolInfo("search", "Search Flyte SDK examples", read_only=True, idempotent=True),
    "search_flyte_docs_examples": ToolInfo("search", "Search Flyte docs examples", read_only=True, idempotent=True),
    "search_full_docs": ToolInfo("search", "Search the full Flyte docs", read_only=True, idempotent=True),
}

ALL_MCP_TOOLS: tuple[MCPTool, ...] = get_args(MCPTool)

#: Tools whose annotations declare ``readOnlyHint=True``; what ``read_only=True`` keeps.
READ_ONLY_MCP_TOOLS: tuple[MCPTool, ...] = tuple(name for name, info in TOOL_REGISTRY.items() if info.read_only)


def _build_group_mapping() -> dict[MCPToolGroup, tuple[MCPTool, ...]]:
    """Derive the group -> tools mapping from `TOOL_REGISTRY`.

    `core` is intentionally empty: the transport endpoints (MCP mount, `/health`) are HTTP
    routes, not MCP "tools".
    """
    mapping: dict[MCPToolGroup, tuple[MCPTool, ...]] = {"all": ALL_MCP_TOOLS, "core": ()}
    for name, info in TOOL_REGISTRY.items():
        mapping[info.group] = (*mapping.get(info.group, ()), name)
    return mapping


TOOL_GROUP_MAPPING: dict[MCPToolGroup, tuple[MCPTool, ...]] = _build_group_mapping()

DEFAULT_IMAGE = (
    flyte.Image.from_debian_base()
    .with_apt_packages("ca-certificates", "git", "curl")
    .with_pip_packages("mcp", "starlette", "uvicorn")
    .with_commands(
        [
            "git clone --depth 1 https://github.com/flyteorg/flyte-sdk.git /root/flyte-sdk",
            "git clone --depth 1 https://github.com/unionai/unionai-examples.git /root/unionai-examples",
            "curl -fsSL https://www.union.ai/docs/v2/union/llms.txt -o /root/llms.txt",
        ]
    )
)


def resolve_tools(
    tool_groups: list[str] | None,
    tools: list[str] | None,
    *,
    read_only: bool = False,
) -> set[str]:
    """Return the set of MCP tool names to expose.

    If both `tool_groups` and `tools` are omitted, all tools are enabled. Otherwise pass
    either one (not both). The `core` group selects no tools; only the HTTP routes are served.

    Args:
        tool_groups: Group names from `TOOL_GROUP_MAPPING`
        tools: Explicit tool names from `ALL_MCP_TOOLS`
        read_only: Drop every tool that is not annotated `readOnlyHint=True`

    Returns:
        The enabled tool names
    """
    if tool_groups is not None and tools is not None:
        raise ValueError("Cannot specify both tool_groups and tools. Choose one.")

    # Declared up front: the branches below produce sets of `MCPTool` literals and of plain
    # `str`, and without this the first branch pins the inferred element type.
    enabled: set[str]
    if tool_groups is None and tools is None:
        enabled = set(ALL_MCP_TOOLS)
    elif tools is not None:
        unknown = [t for t in tools if t not in ALL_MCP_TOOLS]
        if unknown:
            raise ValueError(f"Unknown tool(s): {unknown}. Valid tools: {list(ALL_MCP_TOOLS)}")
        enabled = set(tools)
    else:
        assert tool_groups is not None
        unknown_groups = [g for g in tool_groups if g not in ALL_MCP_TOOL_GROUPS]
        if unknown_groups:
            raise ValueError(f"Unknown tool group(s): {unknown_groups}. Valid groups: {list(ALL_MCP_TOOL_GROUPS)}")
        enabled = set()
        for g in tool_groups:
            enabled.update(TOOL_GROUP_MAPPING[cast(MCPToolGroup, g)])

    if read_only:
        enabled &= set(READ_ONLY_MCP_TOOLS)
    return enabled


def _resolve_tools(tool_groups: list[str] | None, tools: list[str] | None) -> set[str]:
    """Deprecated alias for `resolve_tools`, kept for out-of-tree callers."""
    return resolve_tools(tool_groups, tools)


# ------------------------------
# Environment implementation
# ------------------------------


@dataclass(kw_only=True, repr=True)
class FlyteMCPAppEnvironment(MCPAppEnvironment):
    """Serve a Flyte-facing MCP server over HTTP (FastMCP + Starlette + Uvicorn).

    Use this environment when you want LLM clients to call Flyte operations
    (tasks, runs, actions, logs, apps, triggers, projects, secrets, conditions,
    docs search) through the Model Context Protocol. Install extras with
    `pip install 'flyte[mcp]'`.

    **HTTP layout**

    - `GET /health` — liveness/readiness JSON `{"status": "healthy"}`.
    - The MCP ASGI app is mounted at `mcp_mount_path` (default `/flyte-mcp`). With
      the default `transport="streamable-http"`, the session endpoint is
      `{mcp_mount_path}/mcp` (for example `/flyte-mcp/mcp`). SSE transport uses
      `{mcp_mount_path}/sse` instead.

    **Tool selection**

    Pass `tool_groups` *or* `tools` to restrict which MCP tools are
    registered (not both). Omit both to enable all tools. `read_only=True` then drops
    everything that is not annotated `readOnlyHint=True`, so a public deployment gets a
    safe surface without maintaining a hand-written tool list. Optional allowlists
    limit which tasks, apps, or triggers remote calls may target. Search tools
    require `sdk_examples_path`, `docs_examples_path`, and/or
    `full_docs_path` when those tools are enabled.

    **Project / domain resolution**

    Project- and domain-scoped tools take optional `project`/`domain` arguments. They
    resolve in this order: the explicit argument, then `FLYTE_MCP_PROJECT` /
    `FLYTE_MCP_DOMAIN`, then whatever the initialized config carries. If nothing resolves,
    the tool fails with a message telling the caller to pass them explicitly.

    **Transport security**

    Set `allowed_hosts` / `allowed_origins` (or `FLYTE_MCP_ALLOWED_HOSTS` /
    `FLYTE_MCP_ALLOWED_ORIGINS`) to turn on MCP's DNS-rebinding protection. Any deployment
    reachable over HTTP wants it. When neither is configured the protection stays off,
    preserving the behavior existing deployments were built against.

    **OAuth discovery for MCP clients**

    Set `protected_resource_metadata` to let MCP clients that implement the MCP authorization
    spec discover how to sign in, per RFC 9728, instead of needing a token pasted into their
    configuration. The server then:

    - serves the metadata document, without authentication, at
      `/.well-known/oauth-protected-resource{endpoint path}` (for example
      `/.well-known/oauth-protected-resource/flyte-mcp/mcp`) and at the root
      `/.well-known/oauth-protected-resource`, which clients fall back to;
    - answers any other request without an `Authorization: Bearer` header with `401` and
      `WWW-Authenticate: Bearer resource_metadata="<metadata URL>"`;
    - forwards the bearer token of every other request to Flyte, as with `requires_auth=True`.

    Clients can only reach the metadata if the platform ingress lets unauthenticated requests
    through, so pair it with `requires_auth=False`. The server still refuses the MCP endpoint
    without a bearer token, but it does not validate the token itself: the Flyte endpoint
    validates it on every remote call, so a tool that makes none (the `search` group) answers
    any bearer value.

    ```python
    env = FlyteMCPAppEnvironment(
        name="flyte-mcp",
        requires_auth=False,
        protected_resource_metadata=ProtectedResourceMetadata(),
    )
    ```

    **Image**

    When `image` is omitted (or set to `"auto"`), the environment uses
    `DEFAULT_IMAGE`, which preinstalls the MCP/Starlette/Uvicorn stack
    and clones the flyte-sdk + unionai-examples repos and the Union docs
    `llms.txt` into `/root` so the search tools have content to scan.
    """

    type: str = "FlyteMCPApp"

    title: str | None = None
    instructions: str | None = None

    mcp_mount_path: str = "/flyte-mcp"

    tool_groups: list[str] | None = None
    tools: list[str] | None = None
    read_only: bool = False

    task_allowlist: list[str] | None = None
    app_allowlist: list[str] | None = None
    trigger_allowlist: list[str] | None = None

    # MCP transport security (DNS-rebinding protection)
    allowed_hosts: list[str] | None = None
    allowed_origins: list[str] | None = None

    # RFC 9728 OAuth discovery; see the class docstring
    protected_resource_metadata: ProtectedResourceMetadata | None = None

    # Search default paths
    sdk_examples_path: str | None = "/root/flyte-sdk/examples"
    docs_examples_path: str | None = "/root/unionai-examples/v2"
    full_docs_path: str | None = "/root/llms.txt"

    # The Flyte config file this server was initialized from, as resolved by `flyte-mcp`.
    # It is informational: whoami reports it so a caller can tell which tenant the server
    # is pointed at. Setting it does not change how Flyte is initialized.
    config_file: str | None = None

    mcp: FastMCP | None = field(init=False, default=None)  # type: ignore[assignment]
    _enabled_tools: set[str] = field(init=False, default_factory=set)
    _discovered_authorization_server: str | None = field(init=False, default=None)

    def __post_init__(self):
        if self.tools is not None and self.tool_groups is not None:
            raise ValueError("Cannot specify both tools and tool_groups.")

        if self.protected_resource_metadata is not None and self.transport == "stdio":
            raise ValueError("protected_resource_metadata requires an HTTP transport; stdio has no HTTP surface.")

        if getattr(self, "image", None) in (None, "auto"):
            self.image = DEFAULT_IMAGE

        self._enabled_tools = resolve_tools(self.tool_groups, self.tools, read_only=self.read_only)
        self.mcp = self._create_mcp_server()
        super().__post_init__()
        if self.protected_resource_metadata is not None and self.protected_resource_metadata.resource is None:
            # The request-derived resource ends in this path; reject it once here rather than
            # answering every request with 400.
            _require_resource_path("mcp_mount_path", self._mcp_endpoint_path)

    @property
    def enabled_tools(self) -> set[str]:
        return set(self._enabled_tools)

    # ------------------------------
    # Per-call project / domain
    # ------------------------------

    def resolve_scope(self, project: str | None, domain: str | None) -> tuple[str, str]:
        """Resolve the project/domain a tool call should run against.

        Order: the explicit argument, then the server-wide env default, then the initialized
        config.

        Raises:
            ToolError: when neither is resolvable.
        """
        from flyte._initialize import _get_init_config

        resolved_project = project or os.environ.get(PROJECT_ENV_VAR)
        resolved_domain = domain or os.environ.get(DOMAIN_ENV_VAR)

        cfg = _get_init_config()
        if cfg is not None:
            resolved_project = resolved_project or cfg.project
            resolved_domain = resolved_domain or cfg.domain

        missing = [n for n, v in (("project", resolved_project), ("domain", resolved_domain)) if not v]
        if missing:
            raise ToolError(
                f"Missing {' and '.join(missing)}: pass them as tool arguments or set "
                f"{PROJECT_ENV_VAR}/{DOMAIN_ENV_VAR} on the server."
            )
        return cast(str, resolved_project), cast(str, resolved_domain)

    @contextmanager
    def _scoped(self, project: str | None, domain: str | None) -> Generator[tuple[str, str], None, None]:
        """Run a block with the resolved project/domain installed on the init config.

        Most `flyte.remote` entrypoints (`Run.get`, `Action.listall`, `Secret.*`,
        `Trigger.*`, `Condition.*`, `App.listall`) take no project/domain arguments and
        read them off the init config instead, so scoping the config is the only way to make
        those calls per-call addressable. The override is context-local, so concurrent tool
        calls never see each other's scope.
        """
        from flyte._initialize import _get_init_config, init_config_context

        resolved_project, resolved_domain = self.resolve_scope(project, domain)
        cfg = _get_init_config()
        if cfg is None:
            raise ToolError(
                "Flyte is not initialized on this server, so no remote call can be made. "
                "Start the server with a Flyte config or an API key."
            )
        if cfg.project == resolved_project and cfg.domain == resolved_domain:
            yield resolved_project, resolved_domain
            return
        with init_config_context(cfg.replace(project=resolved_project, domain=resolved_domain)):
            yield resolved_project, resolved_domain

    # ------------------------------
    # Serving
    # ------------------------------

    @property
    def _passthrough_auth(self) -> bool:
        """Whether tool calls act as the caller, using the token on each request.

        Protected-resource metadata implies it even with `requires_auth=False`: that setting
        only stops the ingress from authenticating, so that clients can reach the metadata. The
        server must still require the caller's token and forward it rather than fall back to
        acting as the app's own identity.
        """
        return self.requires_auth or self.protected_resource_metadata is not None

    def _list_metadata_paths(self) -> set[str]:
        """The path-suffixed metadata path for this server's resource, and the root one.

        The root path is served too because clients that got no `resource_metadata` in a
        challenge (an ingress answered instead of this server, or an older client) try it
        after the path-suffixed one. This app has exactly one protected resource, so the root
        document describes it as well.
        """
        prm = self.protected_resource_metadata
        assert prm is not None
        resource_path = urlsplit(prm.resource).path if prm.resource else self._mcp_endpoint_path
        # ASGI decodes the URL path once before routing and middleware matching. Keep
        # the public resource and challenge encoded, but register their decoded path.
        return {unquote(build_metadata_path(resource_path)), WELL_KNOWN_PATH}

    def _resolve_resource_url(self, request: Request) -> str:
        """Prefer the explicit resource over the request origin resolved by the ASGI server."""
        prm = self.protected_resource_metadata
        assert prm is not None
        # INTERNAL_APP_ENDPOINT_PATTERN identifies the private service. It cannot identify
        # the public OAuth resource even when the app itself is publicly accessible.
        resource = prm.resource
        if resource is None:
            # Uvicorn's trusted-proxy configuration owns forwarding-header handling. Reading
            # raw forwarding headers here would let arbitrary clients choose the public URL.
            hosts = request.headers.getlist("host")
            if len(hosts) != 1:
                raise ValueError("Expected exactly one Host header")
            _require_authority("Host", hosts[0])
            # Starlette falls back to the socket address when it cannot use the Host header;
            # that address is never the public resource.
            if request.url.netloc != hosts[0]:
                raise ValueError("Invalid Host header")
            resource = f"{request.url.scheme}://{hosts[0]}{self._mcp_endpoint_path}"
        _require_resource_url("resource", resource)
        return resource

    async def _resolve_authorization_servers(self) -> list[str]:
        prm = self.protected_resource_metadata
        assert prm is not None
        if prm.authorization_servers is not None:
            # Preserve the public list interface, but never publish mutations that bypassed
            # the dataclass's construction-time validation.
            servers = list(prm.authorization_servers)
            try:
                if not servers:
                    raise ValueError("authorization_servers must not be empty.")
                for issuer in servers:
                    _require_issuer_url(issuer)
            except ValueError as e:
                raise _AuthorizationServerDiscoveryError("Configured OAuth issuers are invalid.") from e
            return servers
        # Concurrent first requests may each run the lookup; they get the same answer, so a
        # lock would buy nothing.
        if self._discovered_authorization_server is None:
            issuer = await discover_authorization_server()
            try:
                _require_issuer_url(issuer)
            except ValueError as e:
                raise _AuthorizationServerDiscoveryError(
                    "The Flyte endpoint advertises an invalid OAuth issuer."
                ) from e
            self._discovered_authorization_server = issuer
        return [self._discovered_authorization_server]

    async def _serve_resource_metadata(self, request: Request) -> Response:
        """Serve the RFC 9728 metadata document for this server."""
        from starlette.responses import JSONResponse

        prm = self.protected_resource_metadata
        assert prm is not None
        try:
            resource = self._resolve_resource_url(request)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid resource URL"})
        try:
            authorization_servers = await self._resolve_authorization_servers()
        except (_AuthorizationServerDiscoveryError, ConnectError):
            # Nothing is cached on failure, so the next request retries: a transient
            # control-plane error must not leave the server unable to advertise its
            # authorization server until it restarts.
            logger.exception("Could not discover the authorization server for protected resource metadata")
            return JSONResponse(
                status_code=503,
                content={"detail": "Authorization server discovery failed; see the server logs."},
            )

        # Built as a plain dict rather than through `mcp.shared.auth.ProtectedResourceMetadata`:
        # its `AnyHttpUrl` fields append a slash to a bare-host URL, and clients compare the
        # advertised issuer and resource with the authorization server's `issuer` and their
        # own server URL as strings, so `https://auth.example.com` must not become
        # `https://auth.example.com/`. The URLs were validated when the config was built.
        document = {
            "resource": resource,
            "authorization_servers": authorization_servers,
            "scopes_supported": prm.scopes_supported,
            # MCP only accepts tokens in the Authorization header (no query or form body).
            "bearer_methods_supported": ["header"],
            "resource_name": prm.resource_name,
            "resource_documentation": prm.resource_documentation,
        }
        return JSONResponse(
            {k: v for k, v in document.items() if v is not None},
            # A cache may key only on the URL, ignoring a rewritten Host or ASGI scheme.
            # Only explicitly configured resource identifiers are safe to share.
            headers={"Cache-Control": "public, max-age=3600" if prm.resource is not None else "no-store"},
        )

    def _starlette_routes(self) -> list[BaseRoute]:
        """Serve the protected resource metadata when it is configured."""
        if self.protected_resource_metadata is None:
            return []

        from mcp.server.auth.routes import cors_middleware
        from starlette.routing import Route

        # `mcp.server.auth.routes.create_protected_resource_routes` builds the same route but
        # needs the resource URL up front, which is not known here when it is derived per
        # request. Its CORS wrapper is reused: browser-based clients fetch this cross-origin.
        endpoint = cors_middleware(self._serve_resource_metadata, ["GET", "OPTIONS"])
        return [
            Route(path, endpoint=endpoint, methods=["GET", "OPTIONS"]) for path in sorted(self._list_metadata_paths())
        ]

    def _starlette_middleware(self) -> list[Middleware]:
        """Install the request-scoped auth middleware.

        `FastAPIPassthroughAuthMiddleware` forwards the per-request `Authorization` header to
        Flyte remote calls made against the process-global tenant. `/health` is excluded so
        liveness probes need no credentials. With protected-resource metadata configured,
        `BearerChallengeMiddleware` runs first and turns a request without a bearer token into
        the RFC 9728 challenge, and the metadata paths are excluded from both.
        """
        from starlette.middleware import Middleware

        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        if not self._passthrough_auth:
            return []

        excluded_paths = {"/health"}
        middleware = []
        if self.protected_resource_metadata is not None:
            excluded_paths |= self._list_metadata_paths()
            middleware.append(
                Middleware(
                    BearerChallengeMiddleware,
                    resource_url=self._resolve_resource_url,
                    excluded_paths=excluded_paths,
                    scopes=self.protected_resource_metadata.scopes_supported,
                )
            )
        middleware.append(Middleware(FastAPIPassthroughAuthMiddleware, excluded_paths=excluded_paths))
        return middleware

    async def _starlette_lifespan_startup(self) -> None:
        """Initialize the Flyte SDK in passthrough mode so that Flyte remote calls
        made by tool handlers use the per-request `Authorization` header
        (populated by `FastAPIPassthroughAuthMiddleware`) instead of the
        cluster-injected credentials from `init_in_cluster`.
        """
        project = os.environ.get("FLYTE_PROJECT") or os.environ.get("FLYTE_INTERNAL_EXECUTION_PROJECT")
        domain = os.environ.get("FLYTE_DOMAIN") or os.environ.get("FLYTE_INTERNAL_EXECUTION_DOMAIN")
        if self._passthrough_auth:
            await flyte.init_passthrough.aio(project=project, domain=domain)

    def resolved_allowed_hosts(self) -> list[str] | None:
        """`Host` header allowlist: the explicit field, else `FLYTE_MCP_ALLOWED_HOSTS`."""
        if self.allowed_hosts is not None:
            return list(self.allowed_hosts)
        return _env_csv(ALLOWED_HOSTS_ENV_VAR)

    def resolved_allowed_origins(self) -> list[str] | None:
        """`Origin` header allowlist: the explicit field, else `FLYTE_MCP_ALLOWED_ORIGINS`."""
        if self.allowed_origins is not None:
            return list(self.allowed_origins)
        return _env_csv(ALLOWED_ORIGINS_ENV_VAR)

    def _transport_security_settings(self) -> Any | None:
        """Build `TransportSecuritySettings` for the FastMCP server.

        DNS-rebinding protection is enabled as soon as a host or origin allowlist is configured.
        With protection on, MCP rejects every request whose `Host` is not in `allowed_hosts`,
        and an empty allowlist therefore rejects *everything* — so when neither list is
        configured the protection stays off. That also preserves the behavior existing
        deployments were built against.
        """
        try:  # pragma: no cover - the mcp extra always provides this on supported versions
            from mcp.server.transport_security import TransportSecuritySettings
        except ImportError:
            return None

        hosts = self.resolved_allowed_hosts()
        origins = self.resolved_allowed_origins()

        if not hosts and not origins:
            return TransportSecuritySettings(enable_dns_rebinding_protection=False)

        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=hosts or [],
            allowed_origins=origins or [],
        )

    def _create_mcp_server(self) -> FastMCP:
        try:
            from mcp.server.fastmcp import FastMCP
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "mcp is not installed. Please install 'flyte[mcp]' to use FlyteMCPAppEnvironment."
            ) from e

        transport_security = self._transport_security_settings()

        mcp = FastMCP(
            name=self.title or self.name,
            instructions=self.instructions,
            stateless_http=self.transport == "streamable-http",
            json_response=self.transport == "streamable-http",
            transport_security=transport_security,
        )

        # Explicit conditional registration: disabled tools are never added to the server in
        # the first place, rather than being registered and then popped back out of FastMCP's
        # private ``_tool_manager._tools`` dict.
        tool_functions = build_tool_functions(self)

        undefined = set(TOOL_REGISTRY) - set(tool_functions)
        if undefined:  # pragma: no cover - guards against a registry/implementation drift
            raise RuntimeError(f"Tools declared in TOOL_REGISTRY but not implemented: {sorted(undefined)}")

        for name, fn in tool_functions.items():
            if name not in self._enabled_tools:
                continue
            info = TOOL_REGISTRY[cast(MCPTool, name)]
            mcp.add_tool(fn, name=name, annotations=info.annotations())

        return mcp

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "title", self.title
        yield "type", self.type
        yield "mcp_mount_path", self.mcp_mount_path
        if self.read_only:
            yield "read_only", True
        if self.allowed_hosts is not None:
            yield "allowed_hosts", list(self.allowed_hosts)
        if self.allowed_origins is not None:
            yield "allowed_origins", list(self.allowed_origins)
        if self.protected_resource_metadata is not None:
            yield "protected_resource_metadata", self.protected_resource_metadata
        if self.instructions is not None:
            s = self.instructions
            if len(s) > 80:
                s = s[:77] + "..."
            yield "instructions", s
        if self.tool_groups is not None:
            yield "tool_groups", list(self.tool_groups)
        if self.tools is not None:
            yield "tools", list(self.tools)
        if self.task_allowlist is not None:
            yield "task_allowlist", self.task_allowlist
        if self.app_allowlist is not None:
            yield "app_allowlist", self.app_allowlist
        if self.trigger_allowlist is not None:
            yield "trigger_allowlist", self.trigger_allowlist
