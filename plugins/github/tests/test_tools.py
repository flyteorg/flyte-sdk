"""Tests for the MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from flyteplugins.github import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.github._client import GitHubClient


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
        assert hasattr(GitHubClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions()
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "merge_pull_request" not in fns
    assert "create_issue" not in fns


def test_build_tool_functions_writes():
    fns = build_tool_functions(read_only=False)
    assert "create_issue" in fns
    # destructive tools excluded by default
    assert "merge_pull_request" not in fns
    fns = build_tool_functions(read_only=False, include_destructive=True)
    assert "merge_pull_request" in fns


def test_build_tool_functions_group_filter():
    fns = build_tool_functions(read_only=False, groups=["read"])
    assert all(TOOL_REGISTRY[name].group == "read" for name in fns)


def test_tool_signatures_drop_self():
    fns = build_tool_functions()
    fn = fns["get_pull_request"]
    params = list(inspect.signature(fn).parameters)
    assert params == ["repo", "number"]
    assert fn.__name__ == "get_pull_request"
    assert fn.__doc__


async def test_tool_callable_hits_api(github_api):
    github_api.get("/repos/octo/repo/issues/5").respond(json={"number": 5, "title": "t", "state": "open", "labels": []})
    fns = build_tool_functions(token="t")
    issue = await fns["get_issue"]("octo/repo", 5)
    assert issue["number"] == 5


def test_build_mcp_server_read_only():
    mcp = build_mcp_server()
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    annotations = {t.name: t.annotations for t in tools}
    assert annotations["get_pull_request"].readOnlyHint is True
    assert "merge_pull_request" not in annotations


def test_build_mcp_server_full():
    mcp = build_mcp_server(read_only=False, include_destructive=True)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
    by_name = {t.name: t for t in tools}
    assert by_name["merge_pull_request"].annotations.destructiveHint is True
    assert by_name["create_issue"].annotations.idempotentHint is False
