"""MCP (Model Context Protocol) tool loading for :class:`flyte.ai.agents.Agent`.

This module is internal: import :class:`MCPServerSpec` from
:mod:`flyte.ai.agents` instead. The agent module re-exports the loader for
back-compat with callers that historically imported from
``flyte.ai.agents.agent``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from ._tools import AgentTool

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MCPServerSpec:
    """Declarative spec for a remote MCP server that exposes tools.

    The agent connects on startup, lists available tools, and registers each as
    a callable tool whose ``execute`` proxies the MCP ``tools/call`` request.

    Either ``url`` (for HTTP/SSE/streamable-http transports) or ``command``
    (for stdio transports) must be set.

    Parameters
    ----------
    name:
        Stable display name for logs and event payloads.
    url:
        HTTP(S) URL of the MCP endpoint (e.g. ``https://host/mcp/mcp``).
    command:
        Command to launch a stdio MCP server (e.g.
        ``["uvx", "mcp-server-github"]``).
    headers:
        Optional HTTP headers (for ``Authorization`` etc.).
    env:
        Optional environment variables for stdio launches.
    transport:
        Transport hint. ``"auto"`` (default) infers from ``url`` / ``command``.
    tool_prefix:
        Optional prefix prepended to each tool name to avoid collisions.
    tool_filter:
        Optional allowlist of tool names to expose. ``None`` means all.
    """

    name: str
    url: str | None = None
    command: list[str] | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    transport: Literal["auto", "http", "streamable-http", "sse", "stdio"] = "auto"
    tool_prefix: str = ""
    tool_filter: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.url and not self.command:
            raise ValueError("MCPServerSpec requires either `url` or `command`.")


class _MCPToolLoader:
    """Discovers tools from an MCP server and surfaces them as :class:`AgentTool`.

    Stays inactive until :meth:`load` is called. We delay all MCP imports here
    so that ``Agent`` itself has no required dependency on the ``mcp``
    package.
    """

    def __init__(self, specs: Sequence[MCPServerSpec]):
        self.specs = list(specs)
        self._sessions: list[Any] = []

    async def load(self) -> list[AgentTool]:
        if not self.specs:
            return []
        try:
            from mcp import ClientSession  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP servers configured but the `mcp` package is not installed. "
                "Install with `pip install mcp` or `pip install 'flyte[mcp]'`."
            ) from exc

        tools: list[AgentTool] = []
        for spec in self.specs:
            tools.extend(await self._load_one(spec))
        return tools

    async def _load_one(self, spec: MCPServerSpec) -> list[AgentTool]:
        if spec.command:
            return await self._load_stdio(spec)
        return await self._load_http(spec)

    async def _load_stdio(self, spec: MCPServerSpec) -> list[AgentTool]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert spec.command is not None
        params = StdioServerParameters(command=spec.command[0], args=spec.command[1:], env=spec.env)

        @contextlib.asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

        return await self._materialize(spec, _session)

    async def _load_http(self, spec: MCPServerSpec) -> list[AgentTool]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = spec.url
        assert url is not None
        headers = spec.headers

        @contextlib.asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            async with streamablehttp_client(url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

        return await self._materialize(spec, _session)

    async def _materialize(
        self,
        spec: MCPServerSpec,
        session_cm: Callable[[], "contextlib.AbstractAsyncContextManager[Any]"],
    ) -> list[AgentTool]:
        async with session_cm() as session:
            listing = await session.list_tools()

        tools: list[AgentTool] = []
        for raw_tool in listing.tools:  # type: ignore[attr-defined]
            short_name = getattr(raw_tool, "name", None)
            if not short_name:
                continue
            if spec.tool_filter is not None and short_name not in spec.tool_filter:
                continue
            tool_name = f"{spec.tool_prefix}{short_name}" if spec.tool_prefix else short_name

            description = getattr(raw_tool, "description", "") or f"MCP tool from {spec.name}"
            input_schema = getattr(raw_tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }

            async def _execute(args: dict[str, Any], *, _short_name: str = short_name) -> Any:
                async with session_cm() as inner_session:
                    result = await inner_session.call_tool(_short_name, arguments=args)
                content = getattr(result, "content", None)
                if not content:
                    return None
                if isinstance(content, list):
                    parts: list[Any] = []
                    for chunk in content:
                        text = getattr(chunk, "text", None)
                        parts.append(text if text is not None else chunk)
                    return "\n".join(str(p) for p in parts) if all(isinstance(p, str) for p in parts) else parts
                return content

            tools.append(
                AgentTool(
                    name=tool_name,
                    description=description,
                    parameters=input_schema,
                    execute=_execute,
                    source="mcp",
                )
            )
        return tools
