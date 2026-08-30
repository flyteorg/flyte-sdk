"""Tests for the Notion MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from conftest import page_json, query_response

from flyteplugins.notion import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.notion._client import NotionClient


def test_registry_groups_are_valid():
    for name, info in TOOL_REGISTRY.items():
        assert info.group in ("read", "write"), name
        assert info.title, name
        if info.group == "read":
            assert info.read_only, name
        else:
            assert not info.read_only, name
    assert TOOL_REGISTRY["archive_page"].destructive is True


def test_registry_matches_client_methods():
    for name in TOOL_REGISTRY:
        assert hasattr(NotionClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions(token="k")
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "create_page" not in fns
    assert "archive_page" not in fns


def test_build_tool_functions_destructive_opt_in():
    fns = build_tool_functions(token="k", read_only=False)
    assert "create_page" in fns
    assert "archive_page" not in fns
    fns = build_tool_functions(token="k", read_only=False, include_destructive=True)
    assert "archive_page" in fns


def test_tool_signatures_drop_self():
    fns = build_tool_functions(token="k")
    fn = fns["get_page"]
    assert list(inspect.signature(fn).parameters) == ["page_id"]
    assert fn.__doc__


async def test_tool_callable_hits_api(notion_api):
    notion_api.post("/databases/db1/query").respond(json=query_response([page_json()]))
    fns = build_tool_functions(token="k")
    result = await fns["query_database"]("db1")
    assert result["pages"][0]["id"] == "p1"


def test_build_mcp_server_read_only():
    mcp = build_mcp_server(token="k")
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    assert "archive_page" not in {t.name for t in tools}


def test_build_mcp_server_full():
    mcp = build_mcp_server(token="k", read_only=False, include_destructive=True)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
    by_name = {t.name: t for t in tools}
    assert by_name["archive_page"].annotations.destructiveHint is True
