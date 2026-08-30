"""Tests for the ClickUp MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from flyteplugins.clickup import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.clickup._client import ClickUpClient


def test_registry_groups_are_valid():
    for name, info in TOOL_REGISTRY.items():
        assert info.group in ("read", "write"), name
        assert info.title, name
        if info.group == "read":
            assert info.read_only, name
        else:
            assert not info.read_only, name
    assert TOOL_REGISTRY["delete_task"].destructive is True


def test_registry_matches_client_methods():
    for name in TOOL_REGISTRY:
        assert hasattr(ClickUpClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions(token="k")
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "create_task" not in fns
    assert "delete_task" not in fns


def test_build_tool_functions_destructive_opt_in():
    fns = build_tool_functions(token="k", read_only=False)
    assert "create_task" in fns
    assert "delete_task" not in fns
    fns = build_tool_functions(token="k", read_only=False, include_destructive=True)
    assert "delete_task" in fns


def test_tool_signatures_drop_self():
    fns = build_tool_functions(token="k")
    fn = fns["list_statuses"]
    assert list(inspect.signature(fn).parameters) == ["list_id"]
    assert fn.__doc__


async def test_tool_callable_hits_api(clickup_api):
    clickup_api.get("/team").respond(json={"teams": [{"id": "w1", "name": "Acme"}]})
    fns = build_tool_functions(token="k")
    workspaces = await fns["list_workspaces"]()
    assert workspaces == [{"id": "w1", "name": "Acme", "color": None}]


def test_build_mcp_server_read_only():
    mcp = build_mcp_server(token="k")
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    assert "delete_task" not in {t.name for t in tools}


def test_build_mcp_server_full():
    mcp = build_mcp_server(token="k", read_only=False, include_destructive=True)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
    by_name = {t.name: t for t in tools}
    assert by_name["delete_task"].annotations.destructiveHint is True
