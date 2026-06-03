"""Tests for flyte.ai.agents.agent (Agent harness)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import sys
from unittest.mock import AsyncMock, patch

import pytest

from flyte import TaskEnvironment
from flyte._context import internal_ctx
from flyte.ai.agents import (
    AccessDenied,
    Agent,
    AgentEvent,
    AgentTool,
    ConcurrencyError,
    LLMMessage,
    MCPServerSpec,
    MemoryStore,
    MemoryStoreError,
    agent_progress_cb,
)
from flyte.ai.agents import tool as tool_decorator
from flyte.ai.agents._tools import (
    _callable_short_doc,
    _json_schema_for_callable,
    _make_callable_tool,
    _make_lazy_entity_tool,
)
from flyte.ai.agents.agent import (
    _abbreviate,
    _hitl_approval,
    _resolve_tools,
    _stringify_tool_result,
    _summarize_signature,
)
from flyte.ai.agents.memory import (
    MESSAGES_PATH,
    _ensure_namespace_segment,
    _join_remote_path,
    _memory_storage_root,
    _normalize_raw_data_path,
)
from flyte.ai.agents.protocol import AgentProtocol, AgentResult
from flyte.models import PathRewrite, RawDataPath

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


class TestToolDecorator:
    def test_bare_decorator_on_callable(self):
        @tool_decorator
        def search(query: str) -> str:
            """Search the corpus."""
            return query

        assert isinstance(search, AgentTool)
        assert search.name == "search"
        assert search.description == "Search the corpus."
        assert search.requires_approval is False
        assert "query" in search.parameters["properties"]

    def test_direct_call_sets_requires_approval(self):
        def issue_refund(order_id: str) -> str:
            """Issue a refund."""
            return order_id

        out = tool_decorator(issue_refund, requires_approval=True)
        assert isinstance(out, AgentTool)
        assert out.requires_approval is True
        assert out.description == "Issue a refund."

    def test_parametrized_decorator_overrides_name_and_description(self):
        @tool_decorator(name="refund", description="Custom desc", requires_approval=True)
        def issue_refund(order_id: str) -> str:
            """Issue a refund."""
            return order_id

        assert issue_refund.name == "refund"
        assert issue_refund.description == "Custom desc"
        assert issue_refund.requires_approval is True

    def test_decorator_on_task_produces_task_tool(self):
        env = TaskEnvironment(name="tool_dec_env", image="auto")

        @tool_decorator(requires_approval=True)
        @env.task
        async def issue_refund(order_id: str, amount: float) -> dict:
            """Issue a refund to the customer."""
            return {"order_id": order_id, "amount": amount}

        assert isinstance(issue_refund, AgentTool)
        assert issue_refund.source == "task"
        assert issue_refund.requires_approval is True
        assert issue_refund.description == "Issue a refund to the customer."
        assert "order_id" in issue_refund.parameters["properties"]

    def test_resolves_in_agent_tools_list(self):
        @tool_decorator(requires_approval=True)
        def dangerous(x: int) -> int:
            """Do something sensitive."""
            return x

        out = _resolve_tools([dangerous])
        assert out["dangerous"].requires_approval is True


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
        result = await agent.run.aio("hi", [])
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
        result = await agent.run.aio("add 1 and 2", [])
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
        result = await agent.run.aio("x", [])
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
        result = await agent.run.aio("trigger boom", [])
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
        result = await agent.run.aio("infinite", [])
        assert "max_turns" in result.error
        assert result.attempts == 3

    async def test_llm_error_returns_error_result(self):
        llm = AsyncMock(side_effect=RuntimeError("provider down"))
        agent = Agent(name="t", instructions="I", call_llm=llm)
        result = await agent.run.aio("x", [])
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
            await agent.run.aio("hi", [])
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
        await agent.run.aio("new", history)
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
        result = await agent.run.aio("do it", [])
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
        await agent.run.aio("do it", [])
        assert executed["ran"] is True


# ----------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMemoryStore:
    async def test_default_root_is_temp_dir(self):
        memory = MemoryStore()
        assert memory.root is not None and memory.root.exists()
        assert memory.messages == []

    async def test_explicit_root_auto_loads_messages(self, tmp_path: pathlib.Path):
        seed = MemoryStore(root=tmp_path)
        seed.append({"role": "user", "content": "hello"})
        seed.append({"role": "assistant", "content": "hi"})
        seed.flush_messages_sync()

        restored = MemoryStore(root=tmp_path)
        assert [m["content"] for m in restored.messages] == ["hello", "hi"]

    async def test_flush_messages_async_persists_transcript(self, tmp_path: pathlib.Path):
        seed = MemoryStore(root=tmp_path)
        seed.append({"role": "user", "content": "hello-async"})
        await seed.flush_messages()

        restored = MemoryStore(root=tmp_path)
        assert [m["content"] for m in restored.messages] == ["hello-async"]

    async def test_corrupt_messages_json_logs_and_resets(self, tmp_path: pathlib.Path):
        (tmp_path / "messages.json").write_text("not json", encoding="utf-8")
        memory = MemoryStore(root=tmp_path)
        assert memory.messages == []

    async def test_explicit_messages_take_precedence_over_disk(self, tmp_path: pathlib.Path):
        seed = MemoryStore(root=tmp_path)
        seed.append({"role": "user", "content": "old"})
        seed.flush_messages_sync()

        restored = MemoryStore(root=tmp_path, messages=[{"role": "user", "content": "fresh"}])
        assert [m["content"] for m in restored.messages] == ["fresh"]

    async def test_memory_persists_conversation(self):
        memory = MemoryStore()
        llm = _make_llm([LLMMessage(content="ack", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm, memory=memory)
        await agent.run.aio("hi", [])
        roles = [m["role"] for m in memory.messages]
        assert roles == ["user", "assistant"]
        assert memory.messages[0]["content"] == "hi"
        assert memory.messages[1]["content"] == "ack"

    async def test_memory_prepended_to_next_run(self):
        memory = MemoryStore(
            messages=[{"role": "user", "content": "remember this"}, {"role": "assistant", "content": "noted"}]
        )
        llm = _make_llm([LLMMessage(content="ok", tool_calls=[])])
        agent = Agent(name="t", instructions="I", call_llm=llm, memory=memory)
        await agent.run.aio("hi", [])
        _, _, messages, _ = llm.await_args_list[0].args
        contents = [m["content"] for m in messages]
        assert "remember this" in contents
        assert "noted" in contents


class TestRemotePathHelpers:
    """Unit coverage for the pure remote-path helpers backing keyed stores."""

    def test_join_remote_path_basic(self):
        assert _join_remote_path("base", "a", "b") == "base/a/b"

    def test_join_remote_path_preserves_uri_scheme(self):
        assert _join_remote_path("s3://bucket", "agents", "v0") == "s3://bucket/agents/v0"

    def test_join_remote_path_strips_surrounding_slashes_on_tail(self):
        assert _join_remote_path("base/", "/a/", "/b/") == "base/a/b"

    def test_join_remote_path_ignores_empty_fragments(self):
        assert _join_remote_path("base", "", "a", "") == "base/a"

    def test_join_remote_path_empty_input(self):
        assert _join_remote_path() == ""
        assert _join_remote_path("", "") == ""

    def test_normalize_raw_data_path_applies_path_rewrite(self):
        rewrite = PathRewrite(
            old_prefix="s3://stable-bucket/persistent/",
            new_prefix="/union-persistent-data/",
        )
        assert (
            _normalize_raw_data_path(
                "/union-persistent-data/w6/demo/flytesnacks/development/u1/a0/hash/rd/abc123",
                rewrite,
            )
            == "s3://stable-bucket/persistent/w6/demo/flytesnacks/development/u1/a0/hash/rd/abc123"
        )

    def test_storage_root_remote_uri_anchored_at_bucket(self):
        assert _memory_storage_root("s3://bucket/w6/org/project/domain/u1/a0/hash/rd/abc123") == "s3://bucket"

    def test_storage_root_local_path_unchanged_without_scratch(self):
        assert _memory_storage_root("/tmp/raw") == "/tmp/raw"

    def test_storage_root_local_path_strips_rd_scratch_suffix(self):
        assert _memory_storage_root("/tmp/raw/rd/abc123") == "/tmp/raw"

    def test_storage_root_trailing_slash_normalized(self):
        assert _memory_storage_root("/tmp/raw/rd/abc123/") == "/tmp/raw"

    def test_storage_root_keeps_non_scratch_rd_lookalike(self):
        # ``rd`` only counts as scratch in the ``.../rd/<run_id>`` tail position.
        assert _memory_storage_root("/tmp/rd/data/keep") == "/tmp/rd/data/keep"

    def test_ensure_namespace_segment_valid(self):
        assert _ensure_namespace_segment("my_key", name="key") == "my_key"

    @pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "..", "."])
    def test_ensure_namespace_segment_rejects_unsafe(self, bad: str):
        with pytest.raises(MemoryStoreError):
            _ensure_namespace_segment(bad, name="key")


@pytest.mark.asyncio
class TestMemoryStoreKeyedRemote:
    async def test_remote_path_for_key_uses_raw_data_project_domain_namespace(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            remote_path = MemoryStore.remote_path_for_key(
                key="my_memory",
                org="org",
                project="proj",
                domain="dev",
            )

        assert remote_path == str(
            tmp_path / "raw" / "agents" / "memory-store" / "v0" / "org" / "proj" / "dev" / "my_memory"
        )

    async def test_remote_path_for_key_strips_per_run_rd_suffix(self, tmp_path: pathlib.Path):
        """``flyte.run`` task contexts use ``{run_base_dir}/rd/{random_id}`` as
        raw-data scratch space. Keyed memories must strip that per-run suffix so
        two runs with the same key resolve to the same store.
        """
        first_ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw" / "rd" / "abc123")))
        second_ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw" / "rd" / "def456")))

        with first_ctx:
            first = MemoryStore.remote_path_for_key(key="my_memory", org="org", project="proj", domain="dev")
        with second_ctx:
            second = MemoryStore.remote_path_for_key(key="my_memory", org="org", project="proj", domain="dev")

        expected = str(tmp_path / "raw" / "agents" / "memory-store" / "v0" / "org" / "proj" / "dev" / "my_memory")
        assert first == expected
        assert second == expected

    async def test_remote_path_for_key_uses_remote_bucket_root(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Hosted task runs can expose ``_F_PATH_REWRITE`` as an absolute S3 URI,
        while the runtime raw-data path is backend-relative. The memory store
        must not embed ``s3://...`` inside that backend-relative key.
        """
        monkeypatch.setenv("_F_PATH_REWRITE", "s3://union-oc-production-persistent/->/union-persistent-data/")
        ctx = internal_ctx().new_raw_data_path(
            RawDataPath(path="s3://union-oc-production-demo/w6/demo/flytesnacks/development/u1/a0/hash/rd/abc123")
        )
        with ctx:
            remote_path = MemoryStore.remote_path_for_key(
                key="my_memory", org="demo", project="flytesnacks", domain="development"
            )

        assert remote_path == (
            "s3://union-oc-production-demo/agents/memory-store/v0/demo/flytesnacks/development/my_memory"
        )

    async def test_remote_path_for_key_uses_remote_bucket_root_with_path_rewrite(self):
        ctx = internal_ctx().new_raw_data_path(
            RawDataPath(
                path="s3://union-oc-production-demo/2f/demo/flytesnacks/development/u1/a0/hash/rd/abc123",
                path_rewrite=PathRewrite(
                    old_prefix="s3://stable-bucket/persistent/",
                    new_prefix="/union-persistent-data/",
                ),
            )
        )
        with ctx:
            remote_path = MemoryStore.remote_path_for_key(
                key="my_memory", org="demo", project="flytesnacks", domain="development"
            )

        assert remote_path == (
            "s3://union-oc-production-demo/agents/memory-store/v0/demo/flytesnacks/development/my_memory"
        )

    async def test_remote_path_for_key_mount_prefix_matches_s3_prefix(self, tmp_path: pathlib.Path):
        """Mounted raw-data prefixes must resolve to the same keyed path as the logical S3 URI."""
        rewrite = PathRewrite(
            old_prefix="s3://union-oc-production-demo/",
            new_prefix="/union-persistent-data/",
        )
        s3_ctx = internal_ctx().new_raw_data_path(
            RawDataPath(
                path="s3://union-oc-production-demo/w6/demo/flytesnacks/development/u1/a0/hash/rd/abc123",
                path_rewrite=rewrite,
            )
        )
        mount_ctx = internal_ctx().new_raw_data_path(
            RawDataPath(
                path="/union-persistent-data/w6/demo/flytesnacks/development/u1/a0/hash/rd/def456",
                path_rewrite=rewrite,
            )
        )
        expected = "s3://union-oc-production-demo/agents/memory-store/v0/demo/flytesnacks/development/my_memory"
        with s3_ctx:
            from_s3 = MemoryStore.remote_path_for_key(
                key="my_memory", org="demo", project="flytesnacks", domain="development"
            )
        with mount_ctx:
            from_mount = MemoryStore.remote_path_for_key(
                key="my_memory", org="demo", project="flytesnacks", domain="development"
            )
        assert from_s3 == expected
        assert from_mount == expected

    async def test_create_saves_empty_store_to_deterministic_path(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            memory = await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")

        expected_root = tmp_path / "raw" / "agents" / "memory-store" / "v0" / "org" / "proj" / "dev" / "my_memory"
        assert memory.key == "my_memory"
        assert memory.remote_path == str(expected_root)
        assert (expected_root / "messages.json").exists()

    async def test_create_errors_when_key_path_already_exists(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")
            with pytest.raises(MemoryStoreError, match="already exists"):
                await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")

    async def test_create_errors_when_only_messages_sentinel_exists(self, tmp_path: pathlib.Path):
        """S3-style stores may not have directory marker objects for prefixes.

        The persisted ``messages.json`` file is enough to prove the memory store
        exists and must block ``create`` / satisfy class-level ``exists``.
        """
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            remote_path = pathlib.Path(
                MemoryStore.remote_path_for_key(key="my_memory", org="org", project="proj", domain="dev")
            )
            remote_path.mkdir(parents=True)
            (remote_path / "messages.json").write_text("[]", encoding="utf-8")

            assert await MemoryStore.exists(key="my_memory", org="org", project="proj", domain="dev")
            with pytest.raises(MemoryStoreError, match="already exists"):
                await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")

    async def test_keyed_save_preserves_messages_at_prefix_root(self, tmp_path: pathlib.Path):
        """Regression: fsspec nests ``put(dir, existing_prefix)`` unless both paths end in ``/``."""
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            memory = await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")
            memory.append({"role": "user", "content": "hi"})
            await memory.save.aio()

            memory.append({"role": "assistant", "content": "hey"})
            await memory.save.aio()

            loaded = await MemoryStore.get_or_create.aio(key="my_memory", org="org", project="proj", domain="dev")

        assert [m["content"] for m in loaded.messages] == ["hi", "hey"]
        remote_root = tmp_path / "raw" / "agents" / "memory-store" / "v0" / "org" / "proj" / "dev" / "my_memory"
        assert (remote_root / MESSAGES_PATH).exists()
        assert not any(remote_root.glob("flyte_agent_mem_*"))

    async def test_get_or_create_loads_existing_store(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            created = await MemoryStore.create.aio(
                key="my_memory", org="org", project="proj", domain="dev", keep_versions=True
            )
            await created.write_text.aio("facts/name.txt", "Ada", expected_sha="")
            await created.save.aio()

            loaded = await MemoryStore.get_or_create.aio(
                key="my_memory", org="org", project="proj", domain="dev", keep_versions=True
            )

        assert loaded.key == "my_memory"
        assert loaded.remote_path == created.remote_path
        assert await loaded.read_text.aio("facts/name.txt") == "Ada"
        await loaded.write_text.aio(
            "facts/name.txt", "Ada Lovelace", expected_sha=await loaded.current_sha.aio("facts/name.txt")
        )
        await loaded.save.aio()
        assert len(list((pathlib.Path(loaded.remote_path) / "versions" / "facts%2Fname.txt").glob("*.txt"))) == 2

    async def test_get_or_create_creates_when_missing(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            memory = await MemoryStore.get_or_create.aio(key="new_memory", org="org", project="proj", domain="dev")

        assert memory.key == "new_memory"
        assert pathlib.Path(memory.remote_path).exists()

    async def test_class_exists_checks_keyed_store_only(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            assert not await MemoryStore.exists(key="my_memory", org="org", project="proj", domain="dev")
            await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")
            assert await MemoryStore.exists(key="my_memory", org="org", project="proj", domain="dev")

    async def test_keyed_save_rejects_non_deterministic_override(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            memory = await MemoryStore.create.aio(key="my_memory", org="org", project="proj", domain="dev")
            with pytest.raises(MemoryStoreError, match="deterministic key path"):
                await memory.save.aio(remote_destination=str(tmp_path / "elsewhere"))

    async def test_key_must_be_single_segment(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            with pytest.raises(MemoryStoreError, match="single path segment"):
                await MemoryStore.create.aio(key="../escape", org="org", project="proj", domain="dev")


class TestMemoryStoreSyncApi:
    """``create`` / ``get_or_create`` / ``save`` are sync-by-default via ``syncify``.

    These are plain (non-async) tests so they exercise the synchronous entrypoints
    that ``.aio()`` mirrors; the syncify background loop inherits the raw-data context
    set here through ``call_soon_threadsafe``'s context copy.
    """

    def test_create_save_and_get_or_create_sync(self, tmp_path: pathlib.Path):
        ctx = internal_ctx().new_raw_data_path(RawDataPath(path=str(tmp_path / "raw")))
        with ctx:
            memory = MemoryStore.create(key="sync_key", org="org", project="proj", domain="dev")
            memory.append({"role": "user", "content": "hi"})
            memory.save()

            loaded = MemoryStore.get_or_create(key="sync_key", org="org", project="proj", domain="dev")

        assert loaded.remote_path == memory.remote_path
        assert [m["content"] for m in loaded.messages] == ["hi"]

    def test_path_addressed_io_sync(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        memory.write_text("notes/plan.md", "sync-step")
        assert memory.read_text("notes/plan.md") == "sync-step"
        memory.write_json("data/x.json", {"k": 1})
        assert memory.read_json("data/x.json") == {"k": 1}


@pytest.mark.asyncio
class TestMemoryStorePathAddressedIO:
    async def test_write_and_read_text_roundtrip(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        meta = await memory.write_text.aio("notes/plan.md", "step 1", actor="tester", reason="unit")
        assert meta.path == "notes/plan.md"
        assert meta.bytes == len(b"step 1")
        assert await memory.read_text.aio("notes/plan.md") == "step 1"

    async def test_write_and_read_json_roundtrip(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_json.aio("data/plan.json", {"a": 1, "b": [1, 2]})
        assert await memory.read_json.aio("data/plan.json") == {"a": 1, "b": [1, 2]}

    async def test_read_text_default_when_missing(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        assert await memory.read_text.aio("missing.txt", default="fallback") == "fallback"
        assert await memory.read_json.aio("missing.json", default={"x": 0}) == {"x": 0}

    async def test_list_paths_excludes_internals_and_messages(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("user/a.txt", "a")
        await memory.write_text.aio("user/b.txt", "b")
        memory.append({"role": "user", "content": "hi"})
        memory.flush_messages_sync()

        paths = memory.list_paths()
        assert paths == ["user/a.txt", "user/b.txt"]
        # Prefixed listing scopes correctly.
        assert memory.list_paths("user") == ["user/a.txt", "user/b.txt"]
        # No bookkeeping paths leak through.
        assert all(not p.startswith(("audit/", "meta/", "versions/")) for p in paths)
        assert "messages.json" not in paths

    async def test_list_paths_skips_symlinked_files(self, tmp_path: pathlib.Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("classified", encoding="utf-8")

        root = tmp_path / "root"
        root.mkdir()
        memory = MemoryStore(root=root)
        await memory.write_text.aio("user/a.txt", "a")

        # Plant a symlink that ``list_paths`` must skip rather than expose.
        (root / "user" / "leak.txt").symlink_to(outside / "secret.txt")

        paths = memory.list_paths()
        assert "user/a.txt" in paths
        assert "user/leak.txt" not in paths

    async def test_list_paths_returns_empty_for_missing_prefix(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        assert memory.list_paths("does/not/exist") == []

    async def test_path_traversal_rejected(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        with pytest.raises(MemoryStoreError, match="traversal"):
            await memory.write_text.aio("../escape.txt", "no")
        with pytest.raises(MemoryStoreError, match="must be relative"):
            await memory.write_text.aio("/abs/path.txt", "no")
        with pytest.raises(MemoryStoreError, match="Empty path"):
            await memory.write_text.aio("", "no")

    async def test_symlink_escape_rejected_on_read_and_write(self, tmp_path: pathlib.Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("classified", encoding="utf-8")

        root = tmp_path / "root"
        root.mkdir()
        # The symlink lives *inside* the memory root but points outside —
        # exactly the threat model for a downloaded ``Dir`` carrying a
        # malicious sidechannel.
        (root / "escape.txt").symlink_to(outside / "secret.txt")

        memory = MemoryStore(root=root)
        with pytest.raises(MemoryStoreError, match="outside the memory root"):
            await memory.read_text.aio("escape.txt")
        with pytest.raises(MemoryStoreError, match="outside the memory root"):
            await memory.write_text.aio("escape.txt", "no")
        # The pristine secret on disk is untouched.
        assert (outside / "secret.txt").read_text(encoding="utf-8") == "classified"

    async def test_messages_json_write_blocked(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        with pytest.raises(AccessDenied, match=r"messages\.json"):
            await memory.write_text.aio("messages.json", "{}")

    async def test_internal_prefixes_blocked(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        for path in ("audit/log.jsonl", "meta/x.json", "versions/x/y.txt"):
            with pytest.raises(AccessDenied, match="reserved"):
                await memory.write_text.aio(path, "no")


@pytest.mark.asyncio
class TestMemoryStoreReadOnlyPrefixes:
    async def test_writes_to_read_only_prefix_rejected(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path, read_only_prefixes=("memory/",))
        with pytest.raises(AccessDenied, match="read-only"):
            await memory.write_text.aio("memory/docs.txt", "no")

    async def test_writes_outside_read_only_prefix_allowed(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path, read_only_prefixes=("memory/",))
        await memory.write_text.aio("user/notes.txt", "ok")
        assert await memory.read_text.aio("user/notes.txt") == "ok"


@pytest.mark.asyncio
class TestMemoryStoreConcurrency:
    async def test_expected_sha_match_succeeds(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("a.txt", "v1")
        sha = await memory.current_sha.aio("a.txt")
        await memory.write_text.aio("a.txt", "v2", expected_sha=sha)
        assert await memory.read_text.aio("a.txt") == "v2"

    async def test_expected_sha_mismatch_raises(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("a.txt", "v1")
        with pytest.raises(ConcurrencyError) as exc:
            await memory.write_text.aio("a.txt", "v2", expected_sha="deadbeef")
        assert exc.value.path == "a.txt"
        assert exc.value.expected_sha == "deadbeef"

    async def test_expected_sha_empty_for_create(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("new.txt", "hello", expected_sha="")
        assert await memory.read_text.aio("new.txt") == "hello"

    async def test_current_sha_empty_when_missing(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        assert await memory.current_sha.aio("nope.txt") == ""

    async def test_current_sha_streams_files_without_sidecar(self, tmp_path: pathlib.Path):
        # A file written outside MemoryStore (no metadata sidecar) should
        # still hash correctly through the streaming fallback path.
        root = tmp_path
        (root / "user").mkdir()
        (root / "user" / "raw.txt").write_text("raw-content", encoding="utf-8")
        memory = MemoryStore(root=root)
        expected = hashlib.sha256(b"raw-content").hexdigest()
        assert await memory.current_sha.aio("user/raw.txt") == expected


@pytest.mark.asyncio
class TestMemoryStoreAudit:
    async def test_audit_records_create_then_update(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("a.txt", "v1", actor="alice", reason="seed")
        await memory.write_text.aio("a.txt", "v2", actor="bob", reason="update")
        events = await memory.audit_tail()
        assert len(events) == 2
        assert events[0]["op"] == "create"
        assert events[0]["actor"] == "alice"
        assert events[1]["op"] == "update"
        assert events[1]["old_sha"] == events[0]["new_sha"]

    async def test_audit_disabled_writes_no_log(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path, audit=False)
        await memory.write_text.aio("a.txt", "v1")
        assert await memory.audit_tail() == []
        assert not (tmp_path / "audit" / "log.jsonl").exists()


@pytest.mark.asyncio
class TestMemoryStoreVersions:
    async def test_keep_versions_snapshots_each_write(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path, keep_versions=True)
        await memory.write_text.aio("a.txt", "v1")
        await memory.write_text.aio("a.txt", "v2")
        await memory.write_text.aio("a.txt", "v3")
        snapshots = list((tmp_path / "versions" / "a.txt").glob("*.txt"))
        assert len(snapshots) == 3
        contents = sorted(s.read_text(encoding="utf-8") for s in snapshots)
        assert contents == ["v1", "v2", "v3"]

    async def test_no_versions_by_default(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        await memory.write_text.aio("a.txt", "v1")
        assert not (tmp_path / "versions").exists()


@pytest.mark.asyncio
class TestMemoryStoreMeta:
    async def test_meta_sidecar_records_actor_and_sha(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        meta = await memory.write_text.aio("a.txt", "hello", actor="tool-x", reason="seed")
        loaded = await memory.get_meta.aio("a.txt")
        assert loaded is not None
        assert loaded.path == "a.txt"
        assert loaded.updated_by == "tool-x"
        assert loaded.reason == "seed"
        assert loaded.sha256 == meta.sha256

    async def test_get_meta_none_when_missing(self, tmp_path: pathlib.Path):
        memory = MemoryStore(root=tmp_path)
        assert await memory.get_meta.aio("nope.txt") is None

    async def test_meta_paths_distinct_for_collision_pair(self, tmp_path: pathlib.Path):
        # ``"a/b"`` and ``"a__b"`` would have collided on the old
        # ``replace("/", "__")`` encoding; the URL-quoted encoding keeps
        # their metadata sidecars and version histories distinct.
        memory = MemoryStore(root=tmp_path, keep_versions=True)
        await memory.write_text.aio("a/b", "first")
        await memory.write_text.aio("a__b", "second")

        assert await memory.read_text.aio("a/b") == "first"
        assert await memory.read_text.aio("a__b") == "second"

        meta_slash = await memory.get_meta.aio("a/b")
        meta_under = await memory.get_meta.aio("a__b")
        assert meta_slash is not None and meta_under is not None
        assert meta_slash.sha256 != meta_under.sha256

        # Version histories live in separate encoded directories.
        version_dirs = sorted(p.name for p in (tmp_path / "versions").iterdir() if p.is_dir())
        assert "a%2Fb" in version_dirs
        assert "a__b" in version_dirs


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
        result = await agent.run.aio("hi", [])
        assert result.error == ""

    async def test_mcp_load_called_once(self):
        llm = _make_llm([LLMMessage(content="a", tool_calls=[]), LLMMessage(content="b", tool_calls=[])])
        spec = MCPServerSpec(name="srv", url="https://example.com/mcp")
        agent = Agent(name="t", instructions="I", call_llm=llm, mcp_servers=[spec])

        load_mock = AsyncMock(return_value=[])
        with patch.object(agent._mcp_loader, "load", load_mock):
            await agent.run.aio("hi", [])
            await agent.run.aio("hi again", [])
        assert load_mock.await_count == 1

    async def test_mcp_tools_registered_and_listed_after_run(self):
        llm = _make_llm([LLMMessage(content="done", tool_calls=[])])
        spec = MCPServerSpec(name="srv", url="https://example.com/mcp")
        agent = Agent(name="t", instructions="I", call_llm=llm, mcp_servers=[spec])

        async def exec_fn(args):
            return "ok"

        mcp_tool = AgentTool(
            name="mcp_search",
            description="search via mcp",
            parameters={"type": "object", "properties": {}},
            execute=exec_fn,
            source="mcp",
        )
        with patch.object(agent._mcp_loader, "load", AsyncMock(return_value=[mcp_tool])):
            # MCP tools are loaded lazily, so they are absent until the first run.
            assert "mcp_search" not in {d["name"] for d in agent.tool_descriptions()}
            await agent.run.aio("hi", [])

        names = {d["name"] for d in agent.tool_descriptions()}
        assert "mcp_search" in names
        assert "mcp_search" in agent.system_prompt

    async def test_mcp_tool_name_collision_is_skipped(self):
        # A local tool already named ``_add`` must win over an MCP tool of the
        # same name; the MCP one is skipped with a warning rather than clobbering.
        llm = _make_llm([LLMMessage(content="done", tool_calls=[])])
        spec = MCPServerSpec(name="srv", url="https://example.com/mcp")
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm, mcp_servers=[spec])
        original = agent._registry["_add"]

        async def exec_fn(args):
            return "from-mcp"

        clashing = AgentTool(
            name="_add",
            description="mcp add",
            parameters={"type": "object", "properties": {}},
            execute=exec_fn,
            source="mcp",
        )
        with patch.object(agent._mcp_loader, "load", AsyncMock(return_value=[clashing])):
            await agent.run.aio("hi", [])

        # The local tool is preserved; the MCP impostor did not replace it.
        assert agent._registry["_add"] is original
        assert agent._registry["_add"].source == "function"


# ----------------------------------------------------------------------------
# Parallel vs sequential tool execution
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestParallelToolExecution:
    @staticmethod
    def _two_call_llm() -> AsyncMock:
        return _make_llm(
            [
                LLMMessage(
                    content=None,
                    tool_calls=[
                        {"id": "a", "name": "slow", "arguments": {"x": 1}},
                        {"id": "b", "name": "fast", "arguments": {"x": 2}},
                    ],
                ),
                LLMMessage(content="done", tool_calls=[]),
            ]
        )

    @staticmethod
    def _interleaving_tools(order: list[str]):
        async def slow(x: int) -> str:
            """Slow tool."""
            order.append("slow_start")
            await asyncio.sleep(0.02)
            order.append("slow_end")
            return "slow"

        async def fast(x: int) -> str:
            """Fast tool."""
            order.append("fast_start")
            order.append("fast_end")
            return "fast"

        return slow, fast

    async def test_parallel_interleaves_tool_calls(self):
        order: list[str] = []
        slow, fast = self._interleaving_tools(order)
        agent = Agent(
            name="t",
            instructions="I",
            tools={"slow": slow, "fast": fast},
            call_llm=self._two_call_llm(),
            parallel_tool_calls=True,
        )
        await agent.run.aio("go", [])
        # Under concurrent execution, ``fast`` runs (and finishes) while ``slow``
        # is parked on its sleep.
        assert order == ["slow_start", "fast_start", "fast_end", "slow_end"]

    async def test_sequential_preserves_call_order(self):
        order: list[str] = []
        slow, fast = self._interleaving_tools(order)
        agent = Agent(
            name="t",
            instructions="I",
            tools={"slow": slow, "fast": fast},
            call_llm=self._two_call_llm(),
            parallel_tool_calls=False,
        )
        await agent.run.aio("go", [])
        # Strict ordering: ``slow`` fully completes before ``fast`` begins.
        assert order == ["slow_start", "slow_end", "fast_start", "fast_end"]

    async def test_both_tool_results_threaded_back_to_llm(self):
        order: list[str] = []
        slow, fast = self._interleaving_tools(order)
        llm = self._two_call_llm()
        agent = Agent(
            name="t",
            instructions="I",
            tools={"slow": slow, "fast": fast},
            call_llm=llm,
        )
        await agent.run.aio("go", [])
        _, _, second_messages, _ = llm.await_args_list[1].args
        tool_msgs = {m["name"]: m["content"] for m in second_messages if m["role"] == "tool"}
        assert tool_msgs == {"slow": "slow", "fast": "fast"}


# ----------------------------------------------------------------------------
# Dispatch edge cases
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchEdgeCases:
    async def test_non_dict_arguments_wrapped_in_raw(self):
        captured: dict[str, object] = {}

        async def execute(args):
            captured.update(args)
            return "ok"

        weird = AgentTool(
            name="weird",
            description="d",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        llm = _make_llm(
            [
                LLMMessage(content=None, tool_calls=[{"id": "c1", "name": "weird", "arguments": "not-a-dict"}]),
                LLMMessage(content="done", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[weird], call_llm=llm)
        await agent.run.aio("go", [])
        assert captured == {"_raw": "not-a-dict"}

    async def test_missing_arguments_default_to_empty_dict(self):
        captured: dict[str, object] = {"set": False}

        async def execute(args):
            captured["set"] = True
            captured["args"] = args
            return "ok"

        t = AgentTool(
            name="noargs",
            description="d",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        llm = _make_llm(
            [
                LLMMessage(content=None, tool_calls=[{"id": "c1", "name": "noargs"}]),
                LLMMessage(content="done", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[t], call_llm=llm)
        await agent.run.aio("go", [])
        assert captured["set"] is True
        assert captured["args"] == {}


# ----------------------------------------------------------------------------
# Memory persistence on failure paths
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMemoryPersistenceOnFailure:
    async def test_memory_retains_user_message_when_llm_errors(self):
        memory = MemoryStore()
        llm = AsyncMock(side_effect=RuntimeError("provider down"))
        agent = Agent(name="t", instructions="I", call_llm=llm, memory=memory)
        result = await agent.run.aio("remember me", [])
        assert "LLM call failed" in result.error
        # Even though the turn failed, the user message is committed to memory
        # so a later resume sees the in-flight prompt.
        assert [m["content"] for m in memory.messages] == ["remember me"]

    async def test_memory_retains_messages_when_max_turns_reached(self):
        looping = LLMMessage(
            content=None,
            tool_calls=[{"id": "loop", "name": "_add", "arguments": {"x": 0, "y": 0}}],
        )
        memory = MemoryStore()
        llm = AsyncMock(return_value=looping)
        agent = Agent(name="t", instructions="I", tools=[_add], call_llm=llm, memory=memory, max_turns=2)
        await agent.run.aio("loop forever", [])
        roles = [m["role"] for m in memory.messages]
        assert roles[0] == "user"
        # Assistant + tool messages from the capped turns are persisted too.
        assert "assistant" in roles
        assert "tool" in roles


# ----------------------------------------------------------------------------
# Default HITL approval callback
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDefaultHitlApproval:
    async def test_denies_when_hitl_plugin_missing(self):
        t = AgentTool(
            name="sensitive",
            description="d",
            parameters={"type": "object", "properties": {}},
            execute=AsyncMock(),
            requires_approval=True,
        )
        # Force ``import flyteplugins.hitl`` to raise ImportError.
        with patch.dict(sys.modules, {"flyteplugins.hitl": None}):
            approved = await _hitl_approval(t, {"k": "v"})
        assert approved is False


# ----------------------------------------------------------------------------
# _tools internal fallbacks
# ----------------------------------------------------------------------------


class TestToolInternals:
    def test_callable_short_doc_empty_for_undocumented(self):
        def no_doc(x: int) -> int:
            return x

        assert _callable_short_doc(no_doc) == ""

    def test_undocumented_callable_gets_generated_description(self):
        def mystery(x: int) -> int:
            return x

        t = _make_callable_tool(mystery)
        assert t.description == "Execute mystery"

    def test_json_schema_fallback_when_extraction_fails(self):
        def fn(x: int) -> int:
            return x

        with patch("flyte.models.NativeInterface.from_callable", side_effect=RuntimeError("boom")):
            schema = _json_schema_for_callable(fn)
        assert schema == {"type": "object", "properties": {}, "additionalProperties": True}


@pytest.mark.asyncio
class TestLazyEntityTool:
    async def test_lazy_entity_tool_proxies_to_aio(self):
        calls: dict[str, object] = {}

        class _FakeLazyEntity:
            name = "project/domain/fetch_thing"

            async def aio(self, **kwargs):
                calls["kwargs"] = kwargs
                return {"result": 1}

        t = _make_lazy_entity_tool(_FakeLazyEntity())
        assert t.name == "fetch_thing"
        assert t.source == "remote_task"
        out = await t.execute({"a": 2})
        assert out == {"result": 1}
        assert calls["kwargs"] == {"a": 2}

    async def test_lazy_entity_tool_rename(self):
        class _FakeLazyEntity:
            name = "project/domain/fetch_thing"

            async def aio(self, **kwargs):
                return None

        t = _make_lazy_entity_tool(_FakeLazyEntity(), name="renamed")
        assert t.name == "renamed"

    async def test_lazy_entity_routed_through_resolve_tools(self):
        class _FakeLazyEntity:
            name = "project/domain/remote_fn"

            async def aio(self, **kwargs):
                return None

        lazy = _FakeLazyEntity()
        # ``_is_lazy_entity`` does an ``isinstance`` against the real LazyEntity
        # type; patch it so the resolver routes our fake down the remote-task path.
        with patch("flyte.ai.agents._tools._is_lazy_entity", side_effect=lambda obj: obj is lazy):
            out = _resolve_tools([lazy])
        assert "remote_fn" in out
        assert out["remote_fn"].source == "remote_task"


@pytest.mark.asyncio
class TestAsyncToolExecution:
    async def test_async_tool_awaited_in_loop(self):
        # Exercises the async branch of ``_make_callable_tool.execute``.
        llm = _make_llm(
            [
                LLMMessage(content=None, tool_calls=[{"id": "c1", "name": "_async_double", "arguments": {"x": 21}}]),
                LLMMessage(content="42", tool_calls=[]),
            ]
        )
        agent = Agent(name="t", instructions="I", tools=[_async_double], call_llm=llm)
        await agent.run.aio("double 21", [])
        _, _, second_messages, _ = llm.await_args_list[1].args
        tool_msg = next(m for m in second_messages if m["role"] == "tool")
        assert tool_msg["content"] == "42"


class TestStringifyEdgeCases:
    def test_circular_reference_falls_back_to_str(self):
        circular: dict[str, object] = {}
        circular["self"] = circular
        out = _stringify_tool_result(circular)
        # json.dumps raises ValueError on circular refs even with default=str,
        # so we fall back to the plain ``str()`` representation.
        assert "self" in out
