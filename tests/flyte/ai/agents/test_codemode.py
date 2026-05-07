"""Tests for flyte.ai.agents.codemode."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from flyte import TaskEnvironment
from flyte.ai.agents import codemode
from flyte.ai.agents.codemode import (
    CodeModeAgent,
    _extract_code,
    _load_skills,
    _resolve_tools,
    _tool_registry_key,
)
from flyte.ai.agents.protocol import AgentResult


class TestExtractCode:
    def test_python_fence(self):
        text = "Here is code:\n```python\nx = 1 + 2\n```\nend"
        assert _extract_code(text) == "x = 1 + 2"

    def test_fence_without_language_tag(self):
        text = "```\nhello()\n```"
        assert _extract_code(text) == "hello()"

    def test_no_fence_returns_stripped_text(self):
        assert _extract_code("  bare code  ") == "bare code"

    def test_first_fence_wins(self):
        text = "```python\nfirst\n```\n```python\nsecond\n```"
        assert _extract_code(text) == "first"


class TestResolveTools:
    def test_dict_passthrough_copy(self):
        def f():
            return 1

        d = {"a": f}
        out = _resolve_tools(d)
        assert out == d
        assert out is not d  # dict() makes a shallow copy

    def test_sequence_uses_function_names(self):
        def alpha():
            """A."""
            return 1

        def beta():
            """B."""
            return 2

        out = _resolve_tools([alpha, beta])
        assert set(out) == {"alpha", "beta"}
        assert out["alpha"] is alpha

    def test_duplicate_names_raise(self):
        def dup():
            return 0

        other = dup

        with pytest.raises(ValueError, match="Duplicate tool name"):
            _resolve_tools([dup, other])


class TestToolRegistryKey:
    def test_task_template_uses_func_name(self):
        env = TaskEnvironment(name="reg_env", image="auto")

        @env.task
        async def gamma(x: int) -> int:
            return x

        assert _tool_registry_key(gamma) == "gamma"


class TestLoadSkills:
    def test_strings_only(self):
        assert _load_skills(["a", "b"]) == "a\n\nb"

    def test_path_and_string(self, tmp_path: pathlib.Path):
        p = tmp_path / "skill.md"
        p.write_text("from file")
        assert _load_skills(["inline", p]) == "inline\n\nfrom file"


def _add(x: int, y: int) -> int:
    """Add x and y.

    Longer paragraph ignored for short description.
    """
    return x + y


def _mul(a: int, b: int) -> int:
    """Multiply."""
    return a * b


class TestCodeModeAgentToolDescriptions:
    def test_tool_descriptions_shape(self):
        agent = CodeModeAgent([_add, _mul], call_llm=AsyncMock(), max_retries=0)
        descs = agent.tool_descriptions()
        assert len(descs) == 2
        by_name = {d["name"]: d for d in descs}
        assert "_add" in by_name and "_mul" in by_name
        assert by_name["_add"]["signature"].startswith("_add(") and "x:" in by_name["_add"]["signature"]
        assert by_name["_add"]["description"] == "Add x and y."
        assert by_name["_mul"]["description"] == "Multiply."


class TestCodeModeAgentSystemPrompt:
    def test_task_template_signature_in_prompt(self):
        env = TaskEnvironment(name="prompt_env", image="auto")

        @env.task
        async def ranked_metric(rank: int, label: str) -> str:
            """Ranked lookup."""
            return f"{rank}:{label}"

        agent = CodeModeAgent([ranked_metric], call_llm=AsyncMock(), max_retries=0)
        assert "rank:" in agent.system_prompt and "label:" in agent.system_prompt
        assert "ranked_metric" in agent.system_prompt


class TestCodeModeAgentSystemPromptSkills:
    def test_skills_injected(self, tmp_path: pathlib.Path):
        skill_file = tmp_path / "s.txt"
        skill_file.write_text("SKILL_BODY")
        agent = CodeModeAgent(
            [_add],
            call_llm=AsyncMock(),
            max_retries=0,
            skills=["FIRST", skill_file],
            system_prompt_prefix="PREFIX",
        )
        assert "PREFIX" in agent.system_prompt
        assert "FIRST" in agent.system_prompt
        assert "SKILL_BODY" in agent.system_prompt
        assert "Available functions:" in agent.system_prompt
        assert "_add" in agent.system_prompt or "add" in agent.system_prompt

    def test_default_prefix_when_none(self):
        agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=0)
        assert "You are a helpful assistant" in agent.system_prompt


class TestCodeModeAgentFlyteTools:
    def test_uses_flyte_tools_detection(self):
        env = TaskEnvironment(name="detect_env", image="auto")

        @env.task
        async def remote_tool() -> int:
            return 7

        agent = CodeModeAgent([remote_tool], call_llm=AsyncMock(), max_retries=0)
        assert agent.uses_flyte_tools() is True
        plain = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=0)
        assert plain.uses_flyte_tools() is False


@pytest.mark.asyncio
class TestCodeModeAgentRun:
    async def test_success_dict_result(self):
        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "result = {}"
            orch.return_value = {"summary": "S", "charts": ["chart1"]}
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=0)
            out = await agent.run("hi", [{"role": "user", "content": "prev"}])
            assert isinstance(out, AgentResult)
            assert out.code == "result = {}"
            assert out.summary == "S"
            assert out.charts == ["chart1"]
            assert out.error == ""
            assert out.attempts == 1
            orch.assert_awaited_once()
            _, kwargs = orch.await_args
            assert kwargs["tasks"] == [_add]

    async def test_success_non_dict_result(self):
        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "1"
            orch.return_value = 42
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=0)
            out = await agent.run("q", [])
            assert out.summary == "42"
            assert out.charts == []

    async def test_generation_failure(self):
        with patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen:
            gen.side_effect = RuntimeError("llm down")
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=0)
            out = await agent.run("q", [])
            assert "Code generation failed" in out.error
            assert "llm down" in out.error

    async def test_retry_then_success(self):
        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.side_effect = ["bad", "good"]
            orch.side_effect = [ValueError("sandbox oops"), {"summary": "fixed", "charts": []}]
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=2)
            out = await agent.run("q", [])
            assert out.summary == "fixed"
            assert out.code == "good"
            assert out.attempts == 2
            assert orch.await_count == 2

    async def test_retry_llm_fails_returns_partial_error(self):
        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.side_effect = ["bad", Exception("retry llm")]
            orch.side_effect = ValueError("fail")
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=2)
            out = await agent.run("q", [])
            assert "Retry LLM call failed" in out.error
            assert out.code == "bad"

    async def test_sandbox_exhausted_retries(self):
        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "always"
            orch.side_effect = ValueError("nope")
            agent = CodeModeAgent([_add], call_llm=AsyncMock(), max_retries=1)
            out = await agent.run("q", [])
            assert "Sandbox execution failed" in out.error
            assert "attempt" in out.error.lower()

    async def test_task_template_in_tools_used_for_execution(self):
        env = TaskEnvironment(name="exec_env", image="auto")

        @env.task
        async def fetch_metric(name: str) -> dict[str, float]:
            """Durable fetch."""
            return {"jan": 10.0}

        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "pass"
            orch.return_value = {"summary": "ok"}
            agent = CodeModeAgent(
                [fetch_metric, _add],
                call_llm=AsyncMock(),
                max_retries=0,
            )
            assert "fetch_metric(name:" in agent.system_prompt
            assert "Durable fetch." in agent.system_prompt

            await agent.run("x", [])
            _, kwargs = orch.await_args
            assert list(kwargs["tasks"]) == [fetch_metric, _add]

    async def test_dict_renames_tool_for_llm(self):
        env = TaskEnvironment(name="dict_env", image="auto")

        @env.task
        async def fetch_metric_remote(name: str) -> dict[str, float]:
            """Durable fetch."""
            return {"jan": 10.0}

        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "pass"
            orch.return_value = {"summary": "ok"}
            agent = CodeModeAgent(
                {"fetch_metric": fetch_metric_remote},
                call_llm=AsyncMock(),
                max_retries=0,
            )
            descs = agent.tool_descriptions()
            assert [d["name"] for d in descs] == ["fetch_metric"]
            assert descs[0]["signature"].startswith("fetch_metric(name:")

            await agent.run("x", [])
            _, kwargs = orch.await_args
            assert list(kwargs["tasks"]) == [fetch_metric_remote]


@pytest.mark.asyncio
async def test_generate_code_with_mock_llm():
    """Exercise traced generate_code with an injectable LLM (no network)."""

    async def llm(model: str, system: str, messages: list[dict[str, str]]) -> str:
        return "```python\nanswer = 99\n```"

    out = await codemode.generate_code(llm, "any-model", "system", [])
    assert out == "answer = 99"
