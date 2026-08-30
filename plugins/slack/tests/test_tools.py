"""Tests for the MCP tool registry and server builders."""

from __future__ import annotations

import asyncio
import inspect

from flyteplugins.slack import TOOL_REGISTRY, build_mcp_server, build_tool_functions
from flyteplugins.slack._client import SlackClient


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
        assert hasattr(SlackClient, name), f"registry tool {name} has no client method"


def test_build_tool_functions_read_only_default():
    fns = build_tool_functions(bot_token="xoxb")
    assert all(TOOL_REGISTRY[name].read_only for name in fns)
    assert "post_message" not in fns


def test_build_tool_functions_writes():
    fns = build_tool_functions(bot_token="xoxb", read_only=False)
    assert "post_message" in fns


def test_build_tool_functions_group_filter():
    fns = build_tool_functions(bot_token="xoxb", read_only=False, groups=["read"])
    assert all(TOOL_REGISTRY[name].group == "read" for name in fns)


def test_tool_signatures_drop_self():
    fns = build_tool_functions(bot_token="xoxb")
    fn = fns["get_channel_history"]
    params = list(inspect.signature(fn).parameters)
    assert params[0] == "channel"
    assert fn.__name__ == "get_channel_history"
    assert fn.__doc__


async def test_tool_callable_hits_api(slack_api):
    slack_api.get("/conversations.info").respond(json={"ok": True, "channel": {"id": "C123", "name": "general"}})
    fns = build_tool_functions(bot_token="xoxb")
    channel = await fns["get_channel"]("C123")
    assert channel["name"] == "general"


def test_build_mcp_server_read_only():
    mcp = build_mcp_server(bot_token="xoxb")
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len([i for i in TOOL_REGISTRY.values() if i.read_only])
    annotations = {t.name: t.annotations for t in tools}
    assert annotations["get_channel"].readOnlyHint is True
    assert "post_message" not in annotations


def test_build_mcp_server_full():
    mcp = build_mcp_server(bot_token="xoxb", read_only=False)
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == len(TOOL_REGISTRY)
    by_name = {t.name: t for t in tools}
    assert by_name["post_message"].annotations.idempotentHint is False
