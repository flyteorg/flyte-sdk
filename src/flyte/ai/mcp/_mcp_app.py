from __future__ import annotations

import inspect
import pathlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

import rich.repr

import flyte
import flyte.app
import flyte.remote
from flyte._image import Image
from flyte._resources import Resources
from flyte.app._types import Link
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.applications import Starlette


# NOTE: This module uses `from __future__ import annotations`, which means annotations
# on nested MCP tool functions are stored as strings. FastMCP evaluates those strings
# against this module's globals, so we must ensure the context type is available here.
try:  # pragma: no cover
    from mcp.server.fastmcp import Context as MCPContext
except ModuleNotFoundError:  # pragma: no cover

    class MCPContext:  # type: ignore[no-redef]
        pass


# ------------------------------
# Tool types & group mapping
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
    # app
    "get_app",
    "activate_app",
    "deactivate_app",
    # trigger
    "activate_trigger",
    "deactivate_trigger",
    # build
    "build_image",
    # script
    "build_uv_script_image_remote",
    "run_uv_script_remote",
    "flyte_uv_script_format",
    "flyte_uv_script_example",
    # search
    "search_flyte_sdk_examples",
    "search_flyte_docs_examples",
    "search_full_docs",
]

ALL_MCP_TOOLS: tuple[MCPTool, ...] = get_args(MCPTool)

MCPToolGroup = Literal[
    "all",
    "core",
    "task",
    "run",
    "app",
    "trigger",
    "build",
    "script",
    "search",
]

ALL_MCP_TOOL_GROUPS: tuple[MCPToolGroup, ...] = get_args(MCPToolGroup)

TOOL_GROUP_MAPPING: dict[MCPToolGroup, tuple[MCPTool, ...]] = {
    "all": ALL_MCP_TOOLS,
    # Core group is intentionally empty - the transport endpoints (MCP mount, /health)
    # are HTTP routes, not MCP "tools".
    "core": (),
    "task": ("run_task", "get_task", "list_tasks"),
    "run": ("get_run", "get_run_io", "abort_run", "list_runs", "wait_for_run"),
    "app": ("get_app", "activate_app", "deactivate_app"),
    "trigger": ("activate_trigger", "deactivate_trigger"),
    "build": ("build_image",),
    "script": (
        "build_uv_script_image_remote",
        "run_uv_script_remote",
        "flyte_uv_script_format",
        "flyte_uv_script_example",
    ),
    "search": ("search_flyte_sdk_examples", "search_flyte_docs_examples", "search_full_docs"),
}


def _resolve_tools(tool_groups: list[str] | None, tools: list[str] | None) -> set[str]:
    """Return the set of MCP tool names to expose.

    If both arguments are omitted, all tools are enabled. Otherwise pass either
    ``tool_groups`` or ``tools`` (not both). The ``core`` group selects no tools;
    only the HTTP routes are served.
    """
    if tool_groups is None and tools is None:
        return set(ALL_MCP_TOOLS)

    if tool_groups is not None and tools is not None:
        raise ValueError("Cannot specify both tool_groups and tools. Choose one.")

    if tools is not None:
        unknown = [t for t in tools if t not in ALL_MCP_TOOLS]
        if unknown:
            raise ValueError(f"Unknown tool(s): {unknown}. Valid tools: {list(ALL_MCP_TOOLS)}")
        return set(tools)

    assert tool_groups is not None
    unknown_groups = [g for g in tool_groups if g not in ALL_MCP_TOOL_GROUPS]
    if unknown_groups:
        raise ValueError(f"Unknown tool group(s): {unknown_groups}. Valid groups: {list(ALL_MCP_TOOL_GROUPS)}")

    enabled: set[str] = set()
    for g in tool_groups:
        enabled.update(TOOL_GROUP_MAPPING[g])  # type: ignore[index]
    return enabled


# ------------------------------
# Allowlist helpers
# ------------------------------


def _is_task_allowed(allowlist: list[str] | None, domain: str, project: str, name: str) -> bool:
    if allowlist is None:
        return True

    full_path = f"{domain}/{project}/{name}"
    for allowed in allowlist:
        if allowed == full_path:
            return True
        if "/" not in allowed and allowed == name:
            return True
        if allowed.count("/") == 1 and allowed == f"{project}/{name}":
            return True
    return False


def _is_app_allowed(allowlist: list[str] | None, name: str) -> bool:
    if allowlist is None:
        return True
    return name in allowlist


def _is_trigger_allowed(allowlist: list[str] | None, task_name: str, trigger_name: str) -> bool:
    if allowlist is None:
        return True

    full_path = f"{task_name}/{trigger_name}"
    for allowed in allowlist:
        if allowed == full_path:
            return True
        if "/" not in allowed and allowed == trigger_name:
            return True
    return False


# ------------------------------
# Script templates (tools return these)
# ------------------------------


UV_SCRIPT_FORMAT = """# /// script
# dependencies = [
#   "flyte>=0.0.0",
# ]
# ///

import flyte


env = flyte.TaskEnvironment(
    name="my-script",
    image=flyte.Image.from_uv_script(__file__).with_pip_packages(
        # add your packages here
    ),
)


@env.task
def my_task() -> str:
    return "hello"


if __name__ == "__main__":
    # For remote execution with an MCP tool:
    # 1) build_uv_script_image_remote(script=...)
    # 2) run_uv_script_remote(script=...)
    # (Build with --build first, then run.)
    #
    # For local testing:
    flyte.init_passthrough()
    print(my_task())
""".strip()


UV_SCRIPT_EXAMPLE = """# /// script
# dependencies = [
#   "flyte>=0.0.0",
#   "scikit-learn",
# ]
# ///

import asyncio

import flyte
from sklearn.datasets import load_iris


env = flyte.TaskEnvironment(
    name="iris-example",
    image=flyte.Image.from_uv_script(__file__).with_pip_packages("scikit-learn"),
)


@env.task
def load_data() -> int:
    data = load_iris()
    return len(data.data)


async def main() -> None:
    flyte.init_passthrough()
    n = load_data()
    print(f"Loaded rows: {n}")


if __name__ == "__main__":
    # For MCP usage, build with --build first, then run.
    asyncio.run(main())
""".strip()


# ------------------------------
# Search helper
# ------------------------------


async def _search_files(
    pattern: str,
    path: str,
    *,
    top_n: int = 3,
    before_context_lines: int = 5,
    after_context_lines: int = 5,
) -> str:
    """Search files for ``pattern`` and return Markdown with excerpt blocks.

    Recursively scans up to 5000 files under ``path`` (or reads ``path`` if it is
    a file). Files are ranked by match count; the top ``top_n`` files get merged
    context windows around each hit.
    """
    try:
        p = pathlib.Path(path)
        if not p.exists():
            return f"Error: path does not exist: {path}"

        files: list[pathlib.Path] = []
        if p.is_file():
            files = [p]
        else:
            # avoid pathological scans
            for fp in p.rglob("*"):
                if fp.is_file():
                    files.append(fp)
                if len(files) >= 5000:
                    break

        ranked: list[tuple[int, pathlib.Path, list[str]]] = []
        for fp in files:
            try:
                text = fp.read_text(errors="ignore")
            except Exception:
                continue

            if not text:
                continue
            lines = text.splitlines()
            match_idxs = [i for i, line in enumerate(lines) if pattern in line]
            if not match_idxs:
                continue

            ranked.append((len(match_idxs), fp, lines))

        if not ranked:
            return "No matches found"

        ranked.sort(key=lambda t: t[0], reverse=True)
        blocks: list[str] = []
        for count, fp, lines in ranked[: max(1, top_n)]:
            # collect context windows around each match (merge overlapping)
            match_idxs = [i for i, line in enumerate(lines) if pattern in line]
            windows: list[tuple[int, int]] = []
            for i in match_idxs:
                start = max(0, i - before_context_lines)
                end = min(len(lines), i + after_context_lines + 1)
                windows.append((start, end))
            windows.sort()
            merged: list[tuple[int, int]] = []
            for s, e in windows:
                if not merged or s > merged[-1][1]:
                    merged.append((s, e))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))

            excerpt_lines: list[str] = []
            for s, e in merged[:20]:
                excerpt_lines.extend(lines[s:e])
                excerpt_lines.append("...")  # separator between windows

            excerpt = "\n".join(excerpt_lines).rstrip()
            blocks.append(f"### {fp.name} ({count} matches)\n\n```text\n{excerpt}\n```\n")

        return "\n".join(blocks).rstrip()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ------------------------------
# Environment implementation
# ------------------------------


@dataclass(kw_only=True, repr=True)
class FlyteMCPAppEnvironment(flyte.app.AppEnvironment):
    """Serve a Flyte-facing MCP server over HTTP (FastMCP + Starlette + Uvicorn).

    Use this environment when you want LLM clients to call Flyte operations
    (tasks, runs, apps, triggers, image builds, UV scripts, docs search) through
    the Model Context Protocol. Install extras with ``pip install 'flyte[mcp]'``.

    **HTTP layout**

    - ``GET /health`` — liveness/readiness JSON ``{"status": "healthy"}``.
    - The MCP ASGI app is mounted at ``mcp_mount_path`` (default ``/flyte-mcp``). With
      the default ``transport="streamable-http"``, the session endpoint is
      ``{mcp_mount_path}/mcp`` (for example ``/flyte-mcp/mcp``). SSE transport uses
      ``{mcp_mount_path}/sse`` instead.

    **Tool selection**

    Pass ``tool_groups`` *or* ``tools`` to restrict which MCP tools are
    registered (not both). Omit both to enable all tools. Optional allowlists
    limit which tasks, apps, or triggers remote calls may target. Search tools
    require ``sdk_examples_path``, ``docs_examples_path``, and/or
    ``full_docs_path`` when those tools are enabled.

    The UV script remote build/run tools are placeholders when not backed by a
    remote MCP deployment that implements them.
    """

    type: str = "FlyteMCPApp"

    # Presentation
    title: str | None = None
    instructions: str | None = None

    # MCP/HTTP
    mcp_mount_path: str = "/flyte-mcp"
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"
    uvicorn_config: uvicorn.Config | None = None

    # Tool filtering
    tool_groups: list[str] | None = None
    tools: list[str] | None = None

    # Allowlists
    task_allowlist: list[str] | None = None
    app_allowlist: list[str] | None = None
    trigger_allowlist: list[str] | None = None

    # Search configuration
    sdk_examples_path: str | None = None
    docs_examples_path: str | None = None
    full_docs_path: str | None = None

    # private fields
    _mcp_server: FastMCP | None = field(init=False, default=None)
    _starlette_app: Starlette | None = field(init=False, default=None)
    _caller_frame: inspect.FrameInfo | None = field(init=False, default=None)

    def __post_init__(self):
        # Default image suitable for remote serving.
        if getattr(self, "image", None) in (None, "auto"):
            self.image = Image.from_debian_base().with_pip_packages("mcp", "starlette", "uvicorn")
        if getattr(self, "resources", None) is None:
            self.resources = Resources(cpu=1, memory="512Mi")

        super().__post_init__()

        if self.tools is not None and self.tool_groups is not None:
            raise ValueError("Cannot specify both tools and tool_groups.")

        if self.transport not in ["stdio", "sse", "streamable-http"]:
            raise ValueError("transport must be either 'stdio', 'sse', or 'streamable-http'.")

        # Starlette Mount prefix for HTTP only; FastMCP mount_path/streamable_http_path are set in _create_mcp_server.
        if not isinstance(self.mcp_mount_path, str) or not self.mcp_mount_path.startswith("/"):
            raise ValueError("mcp_mount_path must be an absolute path starting with '/'.")

        # Capture instantiation location (useful for resolver args elsewhere)
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

        self._enabled_tools = _resolve_tools(self.tool_groups, self.tools)
        self._mcp_server = self._create_mcp_server()
        self._starlette_app = self._create_starlette_app()

        # Links for the UI.
        mcp_link = self.mcp_mount_path
        if self.transport == "streamable-http":
            mcp_link = f"{self.mcp_mount_path}/mcp"
        elif self.transport == "sse":
            mcp_link = f"{self.mcp_mount_path}/sse"

        self.links = [
            Link(path=mcp_link, title="MCP Endpoint", is_relative=True),
            Link(path="/health", title="Health", is_relative=True),
            *self.links,
        ]

        # Ensure the app process serves our Starlette instance.
        self._server = self._starlette_app_server

    @property
    def enabled_tools(self) -> set[str]:
        return set(self._enabled_tools)

    def _create_mcp_server(self) -> FastMCP:
        try:
            from mcp.server.fastmcp import FastMCP
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "mcp is not installed. Please install 'flyte[mcp]' to use FlyteMCPAppEnvironment."
            ) from e

        # Optional hardening settings
        transport_security: Any | None = None
        try:  # pragma: no cover
            from mcp.server.transport_security import TransportSecuritySettings

            transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        except Exception:
            transport_security = None

        mcp = FastMCP(
            name=self.title or self.name,
            instructions=self.instructions,
            stateless_http=self.transport == "streamable-http",
            json_response=self.transport == "streamable-http",
            transport_security=transport_security,
        )

        # Register all tools, then filter down to enabled_tools. This avoids depending
        # on internal FastMCP registration APIs.
        @mcp.tool()
        async def run_task(
            domain: str,
            project: str,
            name: str,
            inputs: dict,
            version: str | None = None,
            ctx: MCPContext | None = None,
        ) -> dict:
            if not _is_task_allowed(self.task_allowlist, domain, project, name):
                raise ValueError(f"Task {domain}/{project}/{name} is not allowlisted.")
            tk = flyte.remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version="latest" if version is None else None,
            )  # type: ignore[arg-type]
            r = await flyte.run.aio(tk, **inputs)
            return {"url": r.url, "name": r.name}

        @mcp.tool()
        async def get_task(
            domain: str, project: str, name: str, version: str | None = None, ctx: MCPContext | None = None
        ) -> dict:
            if not _is_task_allowed(self.task_allowlist, domain, project, name):
                raise ValueError(f"Task {domain}/{project}/{name} is not allowlisted.")
            lazy = flyte.remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version="latest" if version is None else None,
            )  # type: ignore[arg-type]
            td = await lazy.fetch.aio()
            return {
                "name": td.name,
                "version": td.version,
                "task_type": td.task_type,
                "required_args": td.required_args,
                "default_input_args": td.default_input_args,
                "cache": {
                    "behavior": td.cache.behavior,
                    "version_override": td.cache.version_override,
                    "serialize": td.cache.serialize,
                },
                "secrets": td.secrets,
            }

        @mcp.tool()
        async def list_tasks(
            project: str | None = None,
            domain: str | None = None,
            limit: int = 100,
            entrypoint: bool | None = None,
            ctx: MCPContext | None = None,
        ) -> list[dict]:
            tasks = flyte.remote.Task.listall(project=project, domain=domain, limit=limit, entrypoint=entrypoint)  # type: ignore[arg-type]
            # Best-effort serialization; remote SDK objects usually provide to_dict().
            out: list[dict] = []
            for t in tasks:
                if hasattr(t, "to_dict"):
                    out.append(t.to_dict())
                else:
                    out.append({"name": getattr(t, "name", None), "version": getattr(t, "version", None)})
            return out

        @mcp.tool()
        async def get_run(name: str, ctx: MCPContext | None = None) -> dict:
            run = await flyte.remote.Run.get.aio(name=name)
            return {"name": run.name, "phase": run.phase, "url": run.url, "done": run.done()}

        @mcp.tool()
        async def wait_for_run(
            name: str, poll_interval_s: float = 2.0, timeout_s: float | None = None, ctx: MCPContext | None = None
        ) -> dict:
            run = await flyte.remote.Run.get.aio(name=name)
            watched = await run.watch.aio(interval=poll_interval_s, timeout=timeout_s)  # type: ignore[attr-defined]
            return {"name": watched.name, "phase": watched.phase, "url": watched.url, "done": watched.done()}

        @mcp.tool()
        async def get_run_io(name: str, ctx: MCPContext | None = None) -> dict:
            run = await flyte.remote.Run.get.aio(name=name)
            inputs = await run.inputs.aio()
            outputs = None
            if run.done():
                try:
                    outputs = await run.outputs.aio()
                except Exception:
                    outputs = None
            inputs_dict = (
                dict(inputs)
                if inputs is not None and hasattr(inputs, "keys")
                else (inputs.to_dict() if hasattr(inputs, "to_dict") else inputs)
            )
            outputs_dict: Any = None
            if outputs is not None:
                outputs_dict = (
                    outputs.named_outputs
                    if hasattr(outputs, "named_outputs")
                    else (outputs.to_dict() if hasattr(outputs, "to_dict") else list(outputs))
                )
            return {"name": run.name, "inputs": inputs_dict, "outputs": outputs_dict}

        @mcp.tool()
        async def abort_run(name: str, ctx: MCPContext | None = None) -> dict:
            run = await flyte.remote.Run.get.aio(name=name)
            await run.abort.aio()
            return {"name": run.name, "aborted": True}

        @mcp.tool()
        async def list_runs(
            task_name: str | None = None, limit: int = 100, ctx: MCPContext | None = None
        ) -> list[dict]:
            runs = flyte.remote.Run.listall(task_name=task_name, limit=limit)  # type: ignore[arg-type]
            out: list[dict] = []
            for r in runs:
                if hasattr(r, "to_dict"):
                    out.append(r.to_dict())
                else:
                    out.append({"name": getattr(r, "name", None), "url": getattr(r, "url", None)})
            return out

        @mcp.tool()
        async def get_app(name: str, ctx: MCPContext | None = None) -> dict:
            if not _is_app_allowed(self.app_allowlist, name):
                raise ValueError(f"App {name} is not allowlisted.")
            app = await flyte.remote.App.get.aio(name=name)
            return (
                app.to_dict()
                if hasattr(app, "to_dict")
                else {"name": getattr(app, "name", None), "endpoint": getattr(app, "endpoint", None)}
            )

        @mcp.tool()
        async def activate_app(name: str, ctx: MCPContext | None = None) -> dict:
            if not _is_app_allowed(self.app_allowlist, name):
                raise ValueError(f"App {name} is not allowlisted.")
            app = await flyte.remote.App.get.aio(name=name)
            activated = await app.activate.aio(wait=True)
            return (
                activated.to_dict()
                if hasattr(activated, "to_dict")
                else {"name": getattr(activated, "name", None), "activated": True}
            )

        @mcp.tool()
        async def deactivate_app(name: str, ctx: MCPContext | None = None) -> dict:
            if not _is_app_allowed(self.app_allowlist, name):
                raise ValueError(f"App {name} is not allowlisted.")
            app = await flyte.remote.App.get.aio(name=name)
            deactivated = await app.deactivate.aio(wait=True)
            return (
                deactivated.to_dict()
                if hasattr(deactivated, "to_dict")
                else {"name": getattr(deactivated, "name", None), "deactivated": True}
            )

        @mcp.tool()
        async def activate_trigger(task_name: str, trigger_name: str, ctx: MCPContext | None = None) -> dict:
            if not _is_trigger_allowed(self.trigger_allowlist, task_name, trigger_name):
                raise ValueError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
            t = await flyte.remote.Trigger.get.aio(task_name=task_name, name=trigger_name)
            activated = await t.activate.aio(wait=True)  # type: ignore[attr-defined]
            return (
                activated.to_dict()
                if hasattr(activated, "to_dict")
                else {"task_name": task_name, "name": trigger_name, "activated": True}
            )

        @mcp.tool()
        async def deactivate_trigger(task_name: str, trigger_name: str, ctx: MCPContext | None = None) -> dict:
            if not _is_trigger_allowed(self.trigger_allowlist, task_name, trigger_name):
                raise ValueError(f"Trigger {task_name}/{trigger_name} is not allowlisted.")
            t = await flyte.remote.Trigger.get.aio(task_name=task_name, name=trigger_name)
            deactivated = await t.deactivate.aio(wait=True)  # type: ignore[attr-defined]
            return (
                deactivated.to_dict()
                if hasattr(deactivated, "to_dict")
                else {"task_name": task_name, "name": trigger_name, "deactivated": True}
            )

        @mcp.tool()
        async def build_image(image: str, ctx: MCPContext | None = None) -> dict:
            # Best-effort hook into existing build machinery.
            # Users can also invoke image builds via the CLI / webhook app; this is a thin wrapper.
            from flyte._internal.imagebuild import remote_builder as _remote_builder

            _build_image = getattr(_remote_builder, "build_image", None)
            if _build_image is None:  # pragma: no cover
                raise NotImplementedError("Image build is not available in this Flyte installation.")

            run = await _build_image(image)  # type: ignore[misc]
            return run if isinstance(run, dict) else {"result": str(run)}

        @mcp.tool()
        async def build_uv_script_image_remote(script: str, ctx: MCPContext | None = None) -> dict:
            raise NotImplementedError(
                "Remote UV script image builds require a remote MCP backend. "
                "Use a connected remote MCP server for this tool."
            )

        @mcp.tool()
        async def run_uv_script_remote(script: str, ctx: MCPContext | None = None) -> dict:
            raise NotImplementedError(
                "Remote UV script runs require a remote MCP backend. Use a connected remote MCP server for this tool."
            )

        @mcp.tool()
        async def flyte_uv_script_format(ctx: MCPContext | None = None) -> str:
            return UV_SCRIPT_FORMAT

        @mcp.tool()
        async def flyte_uv_script_example(ctx: MCPContext | None = None) -> str:
            return UV_SCRIPT_EXAMPLE

        @mcp.tool()
        async def search_flyte_sdk_examples(
            pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 5, after_context_lines: int = 5
        ) -> str:
            if self.sdk_examples_path is None:
                raise ValueError("sdk_examples_path is not configured for this MCP environment.")
            return await _search_files(
                pattern,
                self.sdk_examples_path,
                top_n=3,
                before_context_lines=before_context_lines,
                after_context_lines=after_context_lines,
            )

        @mcp.tool()
        async def search_flyte_docs_examples(
            pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 5, after_context_lines: int = 5
        ) -> str:
            if self.docs_examples_path is None:
                raise ValueError("docs_examples_path is not configured for this MCP environment.")
            return await _search_files(
                pattern,
                self.docs_examples_path,
                top_n=3,
                before_context_lines=before_context_lines,
                after_context_lines=after_context_lines,
            )

        @mcp.tool()
        async def search_full_docs(
            pattern: str, ctx: MCPContext | None = None, before_context_lines: int = 20, after_context_lines: int = 20
        ) -> str:
            if self.full_docs_path is None:
                raise ValueError("full_docs_path is not configured for this MCP environment.")
            return await _search_files(
                pattern,
                self.full_docs_path,
                top_n=3,
                before_context_lines=before_context_lines,
                after_context_lines=after_context_lines,
            )

        # Filter tools down to the enabled set.
        tool_manager = getattr(mcp, "_tool_manager", None)
        if tool_manager is not None and hasattr(tool_manager, "_tools"):
            current = set(tool_manager._tools.keys())
            for name in current:
                if name not in self._enabled_tools:
                    tool_manager._tools.pop(name, None)

        return mcp

    def _create_starlette_app(self) -> Starlette:
        try:
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Mount, Route
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "starlette is not installed. Please install 'flyte[mcp]' to use FlyteMCPAppEnvironment."
            ) from e

        assert self._mcp_server is not None

        async def _health(_: Any) -> JSONResponse:
            return JSONResponse({"status": "healthy"})

        # FastMCP exposes different ASGI apps; pick one consistent with ``transport``.
        mcp_asgi = None
        if self.transport == "sse":
            if hasattr(self._mcp_server, "sse_app"):
                mcp_asgi = self._mcp_server.sse_app()
            else:  # pragma: no cover
                raise RuntimeError("FastMCP does not expose sse_app(); cannot use transport='sse'.")
        elif hasattr(self._mcp_server, "streamable_http_app"):
            mcp_asgi = self._mcp_server.streamable_http_app()
        elif hasattr(self._mcp_server, "sse_app"):  # pragma: no cover
            mcp_asgi = self._mcp_server.sse_app()
        else:  # pragma: no cover
            raise RuntimeError("FastMCP does not expose an ASGI app (expected streamable_http_app or sse_app).")

        routes = [
            Mount(self.mcp_mount_path, app=mcp_asgi),
            Route("/health", endpoint=_health, methods=["GET"]),
        ]

        # The mounted FastMCP app registers a lifespan (starts StreamableHTTPSessionManager).
        # Starlette does not run mounted apps' lifespans unless we compose them here.
        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with mcp_asgi.router.lifespan_context(mcp_asgi):
                yield

        return Starlette(routes=routes, lifespan=lifespan)

    async def _starlette_app_server(self):
        try:
            import uvicorn
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError("uvicorn is not installed. Please install 'flyte[mcp]' to serve this app.") from e

        assert self._starlette_app is not None
        if self.uvicorn_config is None:
            self.uvicorn_config = uvicorn.Config(self._starlette_app, port=self.port.port)
        elif self.uvicorn_config.port is None:
            self.uvicorn_config.port = self.port.port

        await uvicorn.Server(self.uvicorn_config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        # The base AppEnvironment provides `container_cmd`; app extras override this to keep
        # container command empty and let the runtime decide.
        return []

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "title", self.title
        yield "type", self.type
        yield "mcp_mount_path", self.mcp_mount_path
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
