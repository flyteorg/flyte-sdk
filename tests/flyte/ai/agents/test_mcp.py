"""Tests for flyte.ai.agents._mcp (MCP tool discovery + proxying).

The real MCP transports (stdio / streamable-http) are never started here.
Instead we drive :class:`_MCPToolLoader._materialize` with a fake session
context-manager so we can exercise tool filtering, prefixing, schema/description
defaults, and the ``call_tool`` content-normalization logic directly.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest

from flyte.ai.agents import MCPServerSpec
from flyte.ai.agents._mcp import _MCPToolLoader

# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------


class _RawTool:
    def __init__(self, name, description=None, inputSchema=None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class _Listing:
    def __init__(self, tools):
        self.tools = tools


class _Chunk:
    def __init__(self, text):
        self.text = text


class _CallResult:
    def __init__(self, content):
        self.content = content


class _NonText:
    """A content chunk with no ``.text`` attribute."""

    def __repr__(self):
        return "<nontext>"


class _FakeSession:
    def __init__(self, listing: _Listing, call_result: Any = None):
        self._listing = listing
        self._call_result = call_result
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return self._listing

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._call_result


def _cm_for(session: _FakeSession):
    @contextlib.asynccontextmanager
    async def _cm():
        yield session

    return _cm


# ----------------------------------------------------------------------------
# Spec validation
# ----------------------------------------------------------------------------


class TestMCPServerSpec:
    def test_requires_url_or_command(self):
        with pytest.raises(ValueError, match="url"):
            MCPServerSpec(name="bad")

    def test_url_only_ok(self):
        spec = MCPServerSpec(name="ok", url="https://example.com/mcp")
        assert spec.command is None

    def test_command_only_ok(self):
        spec = MCPServerSpec(name="ok", command=["uvx", "server"])
        assert spec.url is None


# ----------------------------------------------------------------------------
# load() dispatch + optional dependency handling
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLoaderLoad:
    async def test_empty_specs_returns_empty_without_importing_mcp(self, monkeypatch: pytest.MonkeyPatch):
        # Even if `mcp` is unavailable, no specs => no import attempted.
        monkeypatch.setitem(sys.modules, "mcp", None)
        loader = _MCPToolLoader([])
        assert await loader.load() == []

    async def test_missing_mcp_package_raises_helpful_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "mcp", None)
        loader = _MCPToolLoader([MCPServerSpec(name="s", url="https://x/mcp")])
        with pytest.raises(ImportError, match="pip install mcp"):
            await loader.load()

    async def test_load_aggregates_across_specs(self):
        loader = _MCPToolLoader([MCPServerSpec(name="a", url="https://a/mcp"), MCPServerSpec(name="b", command=["x"])])
        loader._load_one = AsyncMock(side_effect=[["tool_a"], ["tool_b1", "tool_b2"]])  # type: ignore[assignment]
        out = await loader.load()
        assert out == ["tool_a", "tool_b1", "tool_b2"]
        assert loader._load_one.await_count == 2

    async def test_load_one_dispatches_stdio_when_command_set(self):
        loader = _MCPToolLoader([])
        loader._load_stdio = AsyncMock(return_value=["stdio"])  # type: ignore[assignment]
        loader._load_http = AsyncMock(return_value=["http"])  # type: ignore[assignment]
        out = await loader._load_one(MCPServerSpec(name="s", command=["x"]))
        assert out == ["stdio"]
        loader._load_stdio.assert_awaited_once()
        loader._load_http.assert_not_awaited()

    async def test_load_one_dispatches_http_when_only_url_set(self):
        loader = _MCPToolLoader([])
        loader._load_stdio = AsyncMock(return_value=["stdio"])  # type: ignore[assignment]
        loader._load_http = AsyncMock(return_value=["http"])  # type: ignore[assignment]
        out = await loader._load_one(MCPServerSpec(name="s", url="https://x/mcp"))
        assert out == ["http"]
        loader._load_http.assert_awaited_once()
        loader._load_stdio.assert_not_awaited()


# ----------------------------------------------------------------------------
# _materialize: tool discovery
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaterialize:
    async def test_builds_tools_with_defaults(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp")
        listing = _Listing([_RawTool("alpha", description="does alpha", inputSchema={"type": "object"})])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        assert len(tools) == 1
        t = tools[0]
        assert t.name == "alpha"
        assert t.description == "does alpha"
        assert t.parameters == {"type": "object"}
        assert t.source == "mcp"

    async def test_missing_description_and_schema_get_defaults(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp")
        listing = _Listing([_RawTool("bare", description=None, inputSchema=None)])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        t = tools[0]
        assert t.description == "MCP tool from srv"
        assert t.parameters["type"] == "object"
        assert t.parameters["additionalProperties"] is True

    async def test_nameless_tools_skipped(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp")
        listing = _Listing([_RawTool(None), _RawTool("real")])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        assert [t.name for t in tools] == ["real"]

    async def test_tool_prefix_applied(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp", tool_prefix="gh_")
        listing = _Listing([_RawTool("issues"), _RawTool("prs")])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        assert sorted(t.name for t in tools) == ["gh_issues", "gh_prs"]

    async def test_tool_filter_allowlist(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp", tool_filter=["keep"])
        listing = _Listing([_RawTool("keep"), _RawTool("drop")])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        assert [t.name for t in tools] == ["keep"]

    async def test_filter_matches_short_name_then_prefix_applied(self):
        # tool_filter compares against the *short* (server) name; the prefix is
        # applied only to the resulting tool name.
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp", tool_prefix="p_", tool_filter=["keep"])
        listing = _Listing([_RawTool("keep"), _RawTool("drop")])
        tools = await loader._materialize(spec, _cm_for(_FakeSession(listing)))
        assert [t.name for t in tools] == ["p_keep"]


# ----------------------------------------------------------------------------
# _materialize: execute() content normalization
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaterializedToolExecute:
    async def _single_tool(self, call_result: Any, *, short_name: str = "t"):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp")
        listing = _Listing([_RawTool(short_name)])
        session = _FakeSession(listing, call_result=call_result)
        tools = await loader._materialize(spec, _cm_for(session))
        return tools[0], session

    async def test_joins_text_chunks(self):
        tool, session = await self._single_tool(_CallResult([_Chunk("line1"), _Chunk("line2")]))
        out = await tool.execute({"q": "x"})
        assert out == "line1\nline2"
        # Proxied to the MCP server using the *short* name + arguments.
        assert session.calls == [("t", {"q": "x"})]

    async def test_mixed_content_returns_list(self):
        nontext = _NonText()
        tool, _ = await self._single_tool(_CallResult([_Chunk("a"), nontext]))
        out = await tool.execute({})
        assert out == ["a", nontext]

    async def test_empty_content_returns_none(self):
        tool, _ = await self._single_tool(_CallResult([]))
        assert await tool.execute({}) is None

    async def test_no_content_attr_returns_none(self):
        tool, _ = await self._single_tool(_CallResult(None))
        assert await tool.execute({}) is None

    async def test_non_list_content_passthrough(self):
        tool, _ = await self._single_tool(_CallResult("raw-string"))
        assert await tool.execute({}) == "raw-string"

    async def test_prefixed_tool_calls_with_short_name(self):
        loader = _MCPToolLoader([])
        spec = MCPServerSpec(name="srv", url="https://x/mcp", tool_prefix="gh_")
        listing = _Listing([_RawTool("issues")])
        session = _FakeSession(listing, call_result=_CallResult([_Chunk("ok")]))
        tools = await loader._materialize(spec, _cm_for(session))
        assert tools[0].name == "gh_issues"
        await tools[0].execute({"n": 1})
        # The server is called with the unprefixed short name.
        assert session.calls == [("issues", {"n": 1})]
