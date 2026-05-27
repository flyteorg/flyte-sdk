"""Tests for flyte.ai.agents.agent (Agent harness)."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from flyte import TaskEnvironment
from flyte.ai.agents import (
    Agent,
    AgentEvent,
    AgentMemory,
    AgentTool,
    LLMMessage,
    MCPServerSpec,
    agent_progress_cb,
)
from flyte.ai.agents.agent import (
    _abbreviate,
    _resolve_tools,
    _stringify_tool_result,
    _summarize_signature,
)
from flyte.ai.agents.protocol import AgentProtocol, AgentResult

# ----------------------------------------------------------------------------
# Tool resolution
# ----------------------------------------------------------------------------


def _add(x: int, y: int) -> int:
    """Add two integers.

    Detailed paragraph that should not appear in the short description.
    """
    return x + y


async def _async_double(x: int) -> int:
    """Double a number asynchronously."""
    return x * 2


class TestResolveTools:
    def test_plain_callable_becomes_function_tool(self):
        out = _resolve_tools([_add])
        assert "_add" in out
        tool = out["_add"]
        assert tool.source == "function"
        assert tool.description.startswith("Add two integers")
        assert tool.parameters["type"] == "object"
        assert "x" in tool.parameters["properties"]
        assert "y" in tool.parameters["properties"]

    def test_dict_renames_tool(self):
        out = _resolve_tools({"sum_two": _add})
        assert "sum_two" in out
        assert "_add" not in out
        assert out["sum_two"].description.startswith("Add two integers")

    def test_async_callable_supported(self):
        out = _resolve_tools([_async_double])
        assert "_async_double" in out
        assert out["_async_double"].source == "function"

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate tool name"):
            _resolve_tools([_add, _add])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot turn"):
            _resolve_tools([42])

    def test_pre_built_agent_tool_passes_through(self):
        async def exec_fn(args):
            return None

        my_tool = AgentTool(
            name="custom_tool",
            description="Custom",
            parameters={"type": "object", "properties": {}},
            execute=exec_fn,
        )
        out = _resolve_tools([my_tool])
        assert out["custom_tool"] is my_tool

    def test_pre_built_agent_tool_with_rename(self):
        async def exec_fn(args):
            return None

        my_tool = AgentTool(
            name="custom_tool",
            description="Custom",
            parameters={"type": "object", "properties": {}},
            execute=exec_fn,
        )
        out = _resolve_tools({"renamed": my_tool})
        assert "renamed" in out
        assert out["renamed"].execute is exec_fn

    def test_flyte_task_resolved(self):
        env = TaskEnvironment(name="resolve_env", image="auto")

        @env.task
        async def fetch_metric(name: str) -> int:
            """Fetch a metric by name."""
            return 0

        out = _resolve_tools([fetch_metric])
        assert "fetch_metric" in out
        tool = out["fetch_metric"]
        assert tool.source == "task"
        assert tool.description == "Fetch a metric by name."
        assert "name" in tool.parameters["properties"]


# ----------------------------------------------------------------------------
# Signature + helpers
# ----------------------------------------------------------------------------


class TestSignatureFormatter:
    def test_required_and_optional_args(self):
        tool = AgentTool(
            name="probe",
            description="d",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                "required": ["a"],
            },
            execute=AsyncMock(),
        )
        assert _summarize_signature(tool) == "probe(a: integer, b?: string)"

    def test_no_properties(self):
        tool = AgentTool(name="t", description="d", parameters={}, execute=AsyncMock())
        assert _summarize_signature(tool) == "t()"


class TestStringify:
    def test_none_becomes_empty(self):
        assert _stringify_tool_result(None) == ""

    def test_string_passes_through(self):
        assert _stringify_tool_result("hello") == "hello"

    def test_dict_jsonified(self):
        out = _stringify_tool_result({"k": 1})
        assert json.loads(out) == {"k": 1}

    def test_non_json_fallback(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        # json.dumps falls through default=str, so we still serialize
        assert "weird" in _stringify_tool_result(Weird())


class TestAbbreviate:
    def test_truncates(self):
        out = _abbreviate("x" * 1000, max_chars=10)
        assert out.startswith("x" * 10)
        assert "+990 chars" in out

    def test_short_passthrough(self):
        assert _abbreviate("ok") == "ok"


# ----------------------------------------------------------------------------
# System prompt + tool catalog
# ----------------------------------------------------------------------------


class TestSystemPrompt:
    def test_includes_instructions(self):
        agent = Agent(name="t", instructions="Be a helpful assistant.")
        assert "Be a helpful assistant." in agent.system_prompt

    def test_lists_tools(self):
        agent = Agent(
            name="t",
            instructions="I",
            tools=[_add, _async_double],
        )
        assert "- _add" in agent.system_prompt
        assert "- _async_double" in agent.system_prompt

    def test_skills_inline_and_file(self, tmp_path: pathlib.Path):
        skill_file = tmp_path / "skill.md"
        skill_file.write_text("FROM_FILE")
        agent = Agent(
            name="t",
            instructions="Base",
            skills=["INLINE_TEXT", skill_file],
        )
        assert "INLINE_TEXT" in agent.system_prompt
        assert "FROM_FILE" in agent.system_prompt

    def test_no_tools_shows_placeholder(self):
        agent = Agent(name="t", instructions="I")
        assert "(no tools registered)" in agent.system_prompt


# ----------------------------------------------------------------------------
# Tool descriptions (Agent protocol)
# ----------------------------------------------------------------------------


class TestToolDescriptions:
    def test_descriptions_have_required_keys(self):
        agent = Agent(name="t", instructions="I", tools=[_add])
        descs = agent.tool_descriptions()
        assert len(descs) == 1
        d = descs[0]
        assert d["name"] == "_add"
        assert d["signature"].startswith("_add(")
        assert "Add" in d["description"]

    def test_agent_satisfies_protocol(self):
        agent = Agent(name="t", instructions="I")
        assert isinstance(agent, AgentProtocol)


# ----------------------------------------------------------------------------
# Add tool
# ----------------------------------------------------------------------------


class TestAddTool:
    def test_add_tool_after_construction(self):
        agent = Agent(name="t", instructions="I")
        assert len(agent.tool_descriptions()) == 0
        agent.add_tool(_add)
        names = [d["name"] for d in agent.tool_descriptions()]
        assert names == ["_add"]
        assert "- _add" in agent.system_prompt

    def test_add_tool_rename(self):
        agent = Agent(name="t", instructions="I")
        agent.add_tool(_add, name="sum_two")
        names = [d["name"] for d in agent.tool_descriptions()]
        assert names == ["sum_two"]

    def test_add_tool_duplicate_raises(self):
        agent = Agent(name="t", instructions="I", tools=[_add])
        with pytest.raises(ValueError, match="Duplicate"):
            agent.add_tool(_add)


# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------


def _make_llm(responses: list[LLMMessage]) -> AsyncMock:
    """Build an async mock that returns successive ``LLMMessage`` values."""
    mock = AsyncMock(side_effect=responses)
    return mock


@pytest.mark.asyncio
class TestRunLoop:
    async def test_no_tool_calls_returns_text(self):
        llm = _make_llm([LLMMessage(content="Hello human.", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm)
        result = await agent.run("hi", [])
        assert isinstance(result, AgentResult)
        assert result.summary == "Hello human."
        assert result.error == ""
        assert result.attempts == 1
        assert llm.await_count == 1

    async def test_tool_call_then_final_text(self):
        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "_add", "arguments": {"x": 1, "y": 2}}],
                ),
                LLMMessage(content="The answer is 3.", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm)
        result = await agent.run("add 1 and 2", [])
        assert result.summary == "The answer is 3."
        assert result.attempts == 2

        # Inspect the messages handed to the second LLM call.
        _, _, second_messages, _ = llm.await_args_list[1].args
        roles = [m["role"] for m in second_messages]
        assert "tool" in roles
        tool_msg = next(m for m in second_messages if m["role"] == "tool")
        assert tool_msg["name"] == "_add"
        assert tool_msg["content"] == "3"

    async def test_unknown_tool_returns_error_message(self):
        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "nope", "arguments": {}}],
                ),
                LLMMessage(content="Done.", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm)
        result = await agent.run("x", [])
        assert result.summary == "Done."
        _, _, second_messages, _ = llm.await_args_list[1].args
        tool_msg = next(m for m in second_messages if m["role"] == "tool")
        assert "Unknown tool" in tool_msg["content"]

    async def test_tool_exception_is_surfaced(self):
        def boom(x: int) -> int:
            """Boom!"""
            raise RuntimeError("kaboom")

        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "boom", "arguments": {"x": 1}}],
                ),
                LLMMessage(content="Recovered.", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[boom], call_llm=llm)
        result = await agent.run("trigger boom", [])
        assert result.summary == "Recovered."
        _, _, second_messages, _ = llm.await_args_list[1].args
        tool_msg = next(m for m in second_messages if m["role"] == "tool")
        assert "kaboom" in tool_msg["content"]

    async def test_max_turns_caps_loop(self):
        # Always returns a tool call so the loop never ends naturally.
        looping = LLMMessage(
            content=None,
            tool_calls=[{"id": "loop", "name": "_add", "arguments": {"x": 0, "y": 0}}],
        )
        llm = AsyncMock(return_value=looping)
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm, max_turns=3)
        result = await agent.run("infinite", [])
        assert "max_turns" in result.error
        assert result.attempts == 3

    async def test_llm_error_returns_error_result(self):
        llm = AsyncMock(side_effect=RuntimeError("provider down"))
        agent = Agent(name="t", instructions="I", call_llm=llm)
        result = await agent.run("x", [])
        assert "LLM call failed" in result.error
        assert "provider down" in result.error
        assert result.attempts == 1

    async def test_progress_events_emitted(self):
        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "_add", "arguments": {"x": 1, "y": 2}}],
                ),
                LLMMessage(content="3.", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm)

        events: list[AgentEvent] = []

        async def on_event(event: AgentEvent) -> None:
            events.append(event)

        token = agent_progress_cb.set(on_event)
        try:
            await agent.run("hi", [])
        finally:
            agent_progress_cb.reset(token)

        phases = [e.type for e in events]
        assert phases[0] == "agent_start"
        assert phases[-1] == "agent_end"
        assert "tool_start" in phases
        assert "tool_end" in phases

    async def test_history_is_prepended(self):
        llm = _make_llm([LLMMessage(content="ack", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm)
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        await agent.run("new", history)
        _, _, messages, _ = llm.await_args_list[0].args
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]
        assert messages[0]["content"] == "previous question"
        assert messages[-1]["content"] == "new"


# ----------------------------------------------------------------------------
# HITL approval
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApproval:
    async def test_approval_denied_returns_blocked_message(self):
        async def execute(args):
            raise AssertionError("Should not be executed when denied")

        tool = AgentTool(
            name="sensitive",
            description="touchy",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            requires_approval=True,
        )

        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "sensitive", "arguments": {}}],
                ),
                LLMMessage(content="Backed off.", tool_calls=[]),
            ]
        )

        async def deny(t, args):
            return False

        agent = Agent(
            name="t",
            instructions="I",
            tools=[tool],
            call_llm=llm,
            approval_callback=deny,
        )
        result = await agent.run("do it", [])
        assert result.summary == "Backed off."
        _, _, second, _ = llm.await_args_list[1].args
        tool_msg = next(m for m in second if m["role"] == "tool")
        assert "declined" in tool_msg["content"].lower()

    async def test_approval_granted_runs_tool(self):
        executed: dict[str, bool] = {"ran": False}

        async def execute(args):
            executed["ran"] = True
            return "did it"

        tool = AgentTool(
            name="sensitive",
            description="touchy",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            requires_approval=True,
        )
        llm = _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "sensitive", "arguments": {}}],
                ),
                LLMMessage(content="Did it.", tool_calls=[]),
            ]
        )

        async def approve(t, args):
            return True

        agent = Agent(
            name="t",
            instructions="I",
            tools=[tool],
            call_llm=llm,
            approval_callback=approve,
        )
        await agent.run("do it", [])
        assert executed["ran"] is True


# ----------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentMemory:
    async def test_save_and_load_file(self, tmp_path: pathlib.Path):
        memory = AgentMemory(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            artifacts={"plan": {"step": 1}, "raw_text": "note"},
        )

        # Write the JSON ourselves and then exercise from_json directly so the
        # test does not need to talk to remote storage.
        out = json.loads(memory.to_json())
        assert out["messages"][0]["content"] == "hello"
        assert out["artifacts"]["plan"]["step"] == 1
        roundtrip = AgentMemory.from_json(memory.to_json())
        assert roundtrip.messages == memory.messages
        assert roundtrip.artifacts == memory.artifacts

    async def test_empty_memory_roundtrip(self):
        out = AgentMemory.from_json("")
        assert out.messages == []
        assert out.artifacts == {}

    async def test_memory_persists_conversation(self):
        memory = AgentMemory()
        llm = _make_llm([LLMMessage(content="ack", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm, memory=memory)
        await agent.run("hi", [])
        roles = [m["role"] for m in memory.messages]
        assert roles == ["user", "assistant"]
        assert memory.messages[0]["content"] == "hi"
        assert memory.messages[1]["content"] == "ack"

    async def test_memory_prepended_to_next_run(self):
        memory = AgentMemory(
            messages=[{"role": "user", "content": "remember this"}, {"role": "assistant", "content": "noted"}]
        )
        llm = _make_llm([LLMMessage(content="ok", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm, memory=memory)
        await agent.run("hi", [])
        _, _, messages, _ = llm.await_args_list[0].args
        contents = [m["content"] for m in messages]
        assert "remember this" in contents
        assert "noted" in contents


# ----------------------------------------------------------------------------
# MCP loader (deferred / optional import behavior)
# ----------------------------------------------------------------------------


class TestMCPServerSpec:
    def test_requires_url_or_command(self):
        with pytest.raises(ValueError, match="url"):
            MCPServerSpec(name="bad")

    def test_url_ok(self):
        spec = MCPServerSpec(name="ok", url="https://example.com/mcp")
        assert spec.transport == "auto"
        assert spec.url is not None


@pytest.mark.asyncio
class TestMCPLazyLoad:
    async def test_no_mcp_servers_no_import_required(self):
        llm = _make_llm([LLMMessage(content="ok", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm)
        assert agent.mcp_servers == ()
        result = await agent.run("hi", [])
        assert result.error == ""

    async def test_mcp_load_called_once(self):
        llm = _make_llm([LLMMessage(content="a", tool_calls=[]), LLMMessage(content="b", tool_calls=[])])
        spec = MCPServerSpec(name="srv", url="https://example.com/mcp")
        agent = Agent(name="t", instructions="I", call_llm=llm, mcp_servers=[spec])

        load_mock = AsyncMock(return_value=[])
        with patch.object(agent._mcp_loader, "load", load_mock):
            await agent.run("hi", [])
            await agent.run("hi again", [])
        assert load_mock.await_count == 1
