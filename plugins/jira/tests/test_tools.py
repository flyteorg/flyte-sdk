"""Tests for the Jira MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from conftest import SITE, issue_json

from flyteplugins.jira import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.jira._client import JiraClient


def test_registry_groups_are_valid():
    for name, info in TOOL_REGISTRY.items():
        assert info.group in ("read", "write"), name
        assert info.title, name
        if info.group == "read":
            assert info.read_only, name
        else:
            assert not info.read_only, name
    assert TOOL_REGISTRY["delete_issue"].destructive is True


def test_registry_matches_client_methods():
    for name in TOOL_REGISTRY:
        assert hasattr(JiraClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions(base_url=SITE, email="e", api_token="t")
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "create_issue" not in fns
    assert "delete_issue" not in fns


def test_build_tool_functions_destructive_opt_in():
    fns = build_tool_functions(base_url=SITE, email="e", api_token="t", read_only=False)
    assert "create_issue" in fns
    assert "delete_issue" not in fns
    fns = build_tool_functions(base_url=SITE, email="e", api_token="t", read_only=False, include_destructive=True)
    assert "delete_issue" in fns


def test_tool_signatures_drop_self():
    fns = build_tool_functions(base_url=SITE, email="e", api_token="t")
    fn = fns["get_issue"]
    assert list(inspect.signature(fn).parameters) == ["issue_key"]
    assert fn.__doc__


async def test_tool_callable_hits_api(jira_api):
    jira_api.get("/issue/PROJ-1").respond(json=issue_json())
    fns = build_tool_functions(base_url=SITE, email="bot@acme.com", api_token="t")
    issue = await fns["get_issue"]("PROJ-1")
    assert issue["key"] == "PROJ-1"


def test_build_mcp_server_read_only():
    mcp = build_mcp_server(base_url=SITE, email="e", api_token="t")
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    assert "delete_issue" not in {t.name for t in tools}


def test_build_mcp_server_full():
    mcp = build_mcp_server(base_url=SITE, email="e", api_token="t", read_only=False, include_destructive=True)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
    by_name = {t.name: t for t in tools}
    assert by_name["delete_issue"].annotations.destructiveHint is True
