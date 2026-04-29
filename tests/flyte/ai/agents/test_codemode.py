"""Tests for flyte.ai.agents.codemode."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from flyte.ai.agents import codemode
from flyte.ai.agents.codemode import CodeModeAgent, _extract_code, _load_skills, _resolve_tools
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

    async def test_execution_tools_override(self):
        async def exec_only():
            return 0

        exec_only.__name__ = "exec_only"

        with (
            patch.object(codemode, "generate_code", new_callable=AsyncMock) as gen,
            patch.object(codemode.flyte.sandbox, "orchestrate_local", new_callable=AsyncMock) as orch,
        ):
            gen.return_value = "pass"
            orch.return_value = {"summary": "ok"}
            agent = CodeModeAgent(
                [_add],
                execution_tools=[exec_only],
                call_llm=AsyncMock(),
                max_retries=0,
            )
            await agent.run("x", [])
            _, kwargs = orch.await_args
            assert kwargs["tasks"] == [exec_only]


@pytest.mark.asyncio
async def test_generate_code_with_mock_llm():
    """Exercise traced generate_code with an injectable LLM (no network)."""

    async def llm(model: str, system: str, messages: list[dict[str, str]]) -> str:
        return "```python\nanswer = 99\n```"

    out = await codemode.generate_code(llm, "any-model", "system", [])
    assert out == "answer = 99"
