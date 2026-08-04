from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Literal, cast, get_args

import rich.repr

import flyte
from flyte.ai.mcp._mcp_app import MCPAppEnvironment
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
    "ToolError",
    "ToolInfo",
    "_is_app_allowed",
    "_is_task_allowed",
    "_is_trigger_allowed",
    "_search_files",
    "resolve_tools",
]

logger = logging.getLogger(__name__)

#: Set to a truthy value to make the ``flyte-mcp`` CLI serve in multi-tenant central mode.
CENTRAL_MODE_ENV_VAR = "FLYTE_MCP_CENTRAL"

#: Comma-separated ``Host`` header allowlist for MCP transport security.
ALLOWED_HOSTS_ENV_VAR = "FLYTE_MCP_ALLOWED_HOSTS"

#: Comma-separated ``Origin`` header allowlist for MCP transport security.
ALLOWED_ORIGINS_ENV_VAR = "FLYTE_MCP_ALLOWED_ORIGINS"

#: Server-wide default project/domain for tools that do not get them per call.
PROJECT_ENV_VAR = "FLYTE_MCP_PROJECT"
DOMAIN_ENV_VAR = "FLYTE_MCP_DOMAIN"


def _env_csv(var: str) -> list[str] | None:
    """Read a comma-separated env var into a list, or ``None`` when unset/empty."""
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

    This is the single source of truth behind :data:`TOOL_GROUP_MAPPING`, the ``read_only``
    derivation, and the :class:`~mcp.types.ToolAnnotations` attached at registration time —
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

        ``openWorldHint`` is always False: every tool talks to the caller's own Flyte control
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
    """Derive the group -> tools mapping from :data:`TOOL_REGISTRY`.

    ``core`` is intentionally empty: the transport endpoints (MCP mount, ``/health``) are HTTP
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

    If both ``tool_groups`` and ``tools`` are omitted, all tools are enabled. Otherwise pass
    either one (not both). The ``core`` group selects no tools; only the HTTP routes are served.

    :param tool_groups: Group names from :data:`TOOL_GROUP_MAPPING`
    :param tools: Explicit tool names from :data:`ALL_MCP_TOOLS`
    :param read_only: Drop every tool that is not annotated ``readOnlyHint=True``
    :return: The enabled tool names
    """
    if tool_groups is not None and tools is not None:
        raise ValueError("Cannot specify both tool_groups and tools. Choose one.")

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
    """Deprecated alias for :func:`resolve_tools`, kept for out-of-tree callers."""
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
    ``pip install 'flyte[mcp]'``.

    **HTTP layout**

    - ``GET /health`` — liveness/readiness JSON ``{"status": "healthy"}``.
    - The MCP ASGI app is mounted at ``mcp_mount_path`` (default ``/flyte-mcp``). With
      the default ``transport="streamable-http"``, the session endpoint is
      ``{mcp_mount_path}/mcp`` (for example ``/flyte-mcp/mcp``). SSE transport uses
      ``{mcp_mount_path}/sse`` instead.

    **Tool selection**

    Pass ``tool_groups`` *or* ``tools`` to restrict which MCP tools are
    registered (not both). Omit both to enable all tools. ``read_only=True`` then drops
    everything that is not annotated ``readOnlyHint=True``, so a public deployment gets a
    safe surface without maintaining a hand-written tool list. Optional allowlists
    limit which tasks, apps, or triggers remote calls may target. Search tools
    require ``sdk_examples_path``, ``docs_examples_path``, and/or
    ``full_docs_path`` when those tools are enabled.

    **Project / domain resolution**

    Project- and domain-scoped tools take optional ``project``/``domain`` arguments. They
    resolve in this order: the explicit argument, then ``FLYTE_MCP_PROJECT`` /
    ``FLYTE_MCP_DOMAIN`` (skipped in central mode, where there is no server-wide tenant),
    then whatever the initialized config carries. If nothing resolves, the tool fails with a
    message telling the caller to pass them explicitly.

    **Central (multi-tenant) mode**

    With ``central_mode=True`` a single deployment serves *any* tenant: instead of binding one
    endpoint for the whole process at startup, :class:`~flyte.ai.mcp._tenancy.CentralTenantMiddleware`
    resolves the tenant from each request's ``Authorization: Bearer <api-key>`` and scopes the
    Flyte client to that request. ``requires_auth`` is then irrelevant to tool access — the
    middleware always requires a credential — and no process-global ``init_passthrough`` runs, so
    project/domain must be supplied per call. Restrict which control planes are reachable with
    ``allowed_endpoint_suffixes`` (or ``FLYTE_MCP_ALLOWED_ENDPOINT_SUFFIXES``).

    **Transport security**

    Set ``allowed_hosts`` / ``allowed_origins`` (or ``FLYTE_MCP_ALLOWED_HOSTS`` /
    ``FLYTE_MCP_ALLOWED_ORIGINS``) to turn on MCP's DNS-rebinding protection. When neither is
    configured the protection stays off, preserving the behavior existing per-tenant deployments
    were built against.

    **Image**

    When ``image`` is omitted (or set to ``"auto"``), the environment uses
    :data:`DEFAULT_IMAGE`, which preinstalls the MCP/Starlette/Uvicorn stack
    and clones the flyte-sdk + unionai-examples repos and the Union docs
    ``llms.txt`` into ``/root`` so the search tools have content to scan.
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

    # Multi-tenant central mode
    central_mode: bool = False
    allowed_endpoint_suffixes: list[str] | None = None

    # MCP transport security (DNS-rebinding protection)
    allowed_hosts: list[str] | None = None
    allowed_origins: list[str] | None = None

    # Search default paths
    sdk_examples_path: str | None = "/root/flyte-sdk/examples"
    docs_examples_path: str | None = "/root/unionai-examples/v2"
    full_docs_path: str | None = "/root/llms.txt"

    mcp: FastMCP | None = field(init=False, default=None)  # type: ignore[assignment]
    _enabled_tools: set[str] = field(init=False, default_factory=set)

    def __post_init__(self):
        if self.tools is not None and self.tool_groups is not None:
            raise ValueError("Cannot specify both tools and tool_groups.")

        if self.central_mode and self.transport != "streamable-http":
            # Stateful transports run every tool call for a session in the context of whoever
            # opened it, so a second tenant's credential would be verified and then ignored.
            raise ValueError(
                f"central_mode requires transport='streamable-http' (got {self.transport!r}): "
                "stateful transports bind tool execution to the session opener's tenant."
            )

        if getattr(self, "image", None) in (None, "auto"):
            self.image = DEFAULT_IMAGE

        self._enabled_tools = resolve_tools(self.tool_groups, self.tools, read_only=self.read_only)
        self.mcp = self._create_mcp_server()
        super().__post_init__()

    @property
    def enabled_tools(self) -> set[str]:
        return set(self._enabled_tools)

    # ------------------------------
    # Per-call project / domain
    # ------------------------------

    def resolve_scope(self, project: str | None, domain: str | None) -> tuple[str, str]:
        """Resolve the project/domain a tool call should run against.

        Order: the explicit argument, then the server-wide env default (never in central mode,
        where the process has no tenant of its own), then the initialized config.

        :raises ToolError: when neither is resolvable — in central mode that is the normal
            outcome for a caller who forgot to pass them, so the message says so.
        """
        from flyte._initialize import _get_init_config

        resolved_project = project
        resolved_domain = domain

        if not self.central_mode:
            resolved_project = resolved_project or os.environ.get(PROJECT_ENV_VAR)
            resolved_domain = resolved_domain or os.environ.get(DOMAIN_ENV_VAR)

        cfg = _get_init_config()
        if cfg is not None:
            resolved_project = resolved_project or cfg.project
            resolved_domain = resolved_domain or cfg.domain

        missing = [n for n, v in (("project", resolved_project), ("domain", resolved_domain)) if not v]
        if missing:
            if self.central_mode:
                raise ToolError(
                    f"Missing {' and '.join(missing)}: pass project/domain explicitly — "
                    f"this server is multi-tenant and has no default tenant."
                )
            raise ToolError(
                f"Missing {' and '.join(missing)}: pass them as tool arguments or set "
                f"{PROJECT_ENV_VAR}/{DOMAIN_ENV_VAR} on the server."
            )
        return cast(str, resolved_project), cast(str, resolved_domain)

    @contextmanager
    def _scoped(self, project: str | None, domain: str | None) -> Generator[tuple[str, str], None, None]:
        """Run a block with the resolved project/domain installed on the init config.

        Most ``flyte.remote`` entrypoints (``Run.get``, ``Action.listall``, ``Secret.*``,
        ``Trigger.*``, ``Condition.*``, ``App.listall``) take no project/domain arguments and
        read them off the init config instead, so scoping the config is the only way to make
        those calls per-call addressable. The override is context-local, so concurrent tool
        calls (and concurrent tenants, in central mode) never see each other's scope.
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

    def _starlette_middleware(self) -> list[Middleware]:
        """Install the request-scoped auth middleware.

        In central mode that is :class:`~flyte.ai.mcp._tenancy.CentralTenantMiddleware`, which
        resolves a *different* Flyte tenant per request from the presented credential. It is
        installed regardless of ``requires_auth``, because a central deployment deliberately runs
        with platform auth off (the hosting org's SSO must not gate customer traffic) and relies
        on this middleware for authentication instead.

        Otherwise it is ``FastAPIPassthroughAuthMiddleware``, which forwards the per-request
        ``Authorization`` header to Flyte remote calls made against the single process-global
        tenant. Either way ``/health`` is excluded so liveness probes need no credentials.
        """
        from starlette.middleware import Middleware

        if self.central_mode:
            from flyte.ai.mcp._tenancy import CentralTenantMiddleware

            return [
                Middleware(
                    CentralTenantMiddleware,
                    allowed_endpoint_suffixes=self.allowed_endpoint_suffixes,
                    excluded_paths={"/health"},
                ),
            ]

        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        if not self.requires_auth:
            return []

        return [
            Middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/health"}),
        ]

    async def _starlette_lifespan_startup(self) -> None:
        """Initialize the Flyte SDK in passthrough mode so that Flyte remote calls
        made by tool handlers use the per-request ``Authorization`` header
        (populated by :class:`FastAPIPassthroughAuthMiddleware`) instead of the
        cluster-injected credentials from ``init_in_cluster``.

        Skipped entirely in central mode: there is no process-global tenant to initialize, and
        installing one would give unauthenticated code paths a live client to somebody's control
        plane. ``CentralTenantMiddleware`` supplies the config per request instead.
        """
        if self.central_mode:
            return

        project = os.environ.get("FLYTE_PROJECT") or os.environ.get("FLYTE_INTERNAL_EXECUTION_PROJECT")
        domain = os.environ.get("FLYTE_DOMAIN") or os.environ.get("FLYTE_INTERNAL_EXECUTION_DOMAIN")
        if self.requires_auth:
            await flyte.init_passthrough.aio(project=project, domain=domain)

    def resolved_allowed_hosts(self) -> list[str] | None:
        """``Host`` header allowlist: the explicit field, else ``FLYTE_MCP_ALLOWED_HOSTS``."""
        if self.allowed_hosts is not None:
            return list(self.allowed_hosts)
        return _env_csv(ALLOWED_HOSTS_ENV_VAR)

    def resolved_allowed_origins(self) -> list[str] | None:
        """``Origin`` header allowlist: the explicit field, else ``FLYTE_MCP_ALLOWED_ORIGINS``."""
        if self.allowed_origins is not None:
            return list(self.allowed_origins)
        return _env_csv(ALLOWED_ORIGINS_ENV_VAR)

    def _transport_security_settings(self) -> Any | None:
        """Build ``TransportSecuritySettings`` for the FastMCP server.

        DNS-rebinding protection is enabled as soon as a host or origin allowlist is configured.
        With protection on, MCP rejects every request whose ``Host`` is not in ``allowed_hosts``,
        and an empty allowlist therefore rejects *everything* — so when neither list is
        configured the protection stays off. That also preserves the behavior existing
        per-tenant deployments were built against. In central mode the missing allowlist is
        logged as a warning, since a public endpoint should be pinned to its own hostname.
        """
        try:  # pragma: no cover - the mcp extra always provides this on supported versions
            from mcp.server.transport_security import TransportSecuritySettings
        except Exception:
            return None

        hosts = self.resolved_allowed_hosts()
        origins = self.resolved_allowed_origins()

        if not hosts and not origins:
            if self.central_mode:
                logger.warning(
                    "central_mode is on but no allowed_hosts/allowed_origins were configured "
                    f"(set {ALLOWED_HOSTS_ENV_VAR}); DNS-rebinding protection stays disabled."
                )
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
        if self.central_mode:
            yield "central_mode", True
        if self.read_only:
            yield "read_only", True
        if self.allowed_endpoint_suffixes is not None:
            yield "allowed_endpoint_suffixes", list(self.allowed_endpoint_suffixes)
        if self.allowed_hosts is not None:
            yield "allowed_hosts", list(self.allowed_hosts)
        if self.allowed_origins is not None:
            yield "allowed_origins", list(self.allowed_origins)
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
