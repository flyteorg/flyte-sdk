"""Tests for the Linear MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from conftest import GRAPHQL_URL

from flyteplugins.linear import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.linear._client import LinearClient


def test_registry_groups_are_valid():
    for name, info in TOOL_REGISTRY.items():
        assert info.group in ("read", "write"), name
        assert info.title, name
        if info.group == "read":
            assert info.read_only, name
        else:
            assert not info.read_only, name


def test_registry_matches_client_methods():
    for name in TOOL_REGISTRY:
        assert hasattr(LinearClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions(api_key="k")
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "create_issue" not in fns


def test_build_tool_functions_writes():
    fns = build_tool_functions(api_key="k", read_only=False)
    assert "create_issue" in fns
    assert "add_comment" in fns


def test_tool_signatures_drop_self():
    fns = build_tool_functions(api_key="k")
    fn = fns["get_issue"]
    assert list(inspect.signature(fn).parameters) == ["identifier"]
    assert fn.__doc__


async def test_tool_callable_hits_api(linear_api):
    linear_api.post(GRAPHQL_URL).respond(
        json={"data": {"teams": {"nodes": [{"id": "t1", "key": "ENG", "name": "Eng"}]}}}
    )
    fns = build_tool_functions(api_key="k")
    teams = await fns["list_teams"]()
    assert teams[0]["key"] == "ENG"


def test_build_mcp_server_read_only():
    mcp = build_mcp_server(api_key="k")
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    names = {t.name for t in tools}
    assert "create_issue" not in names


def test_build_mcp_server_full():
    mcp = build_mcp_server(api_key="k", read_only=False)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
