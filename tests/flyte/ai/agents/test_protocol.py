"""Tests for flyte.ai.agents.protocol."""

from dataclasses import fields

import pytest

from flyte.ai.agents.protocol import Agent, AgentResult


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult()
        assert r.code == ""
        assert r.charts == []
        assert r.summary == ""
        assert r.error == ""
        assert r.attempts == 1

    def test_field_mutability_charts_is_new_list_per_instance(self):
        a = AgentResult()
        b = AgentResult()
        a.charts.append("x")
        assert b.charts == []

    def test_explicit_values(self):
        r = AgentResult(
            code="print(1)",
            charts=["data:image/png;base64,abc"],
            summary="Done",
            error="",
            attempts=3,
        )
        assert r.code == "print(1)"
        assert r.charts == ["data:image/png;base64,abc"]
        assert r.summary == "Done"
        assert r.attempts == 3

    def test_dataclass_fields(self):
        names = {f.name for f in fields(AgentResult)}
        assert names == {"code", "charts", "summary", "error", "attempts"}


class _MinimalAgent:
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        return AgentResult(summary=message)

    def tool_descriptions(self) -> list[dict[str, str]]:
        return [{"name": "t", "signature": "t()", "description": "d"}]


class _MissingToolDescriptions:
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        return AgentResult()


class TestAgentProtocol:
    @pytest.mark.asyncio
    async def test_runtime_checkable_positive(self):
        agent = _MinimalAgent()
        assert isinstance(agent, Agent)

    def test_runtime_checkable_negative_missing_method(self):
        class Bad:
            async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
                return AgentResult()

        assert not isinstance(Bad(), Agent)

    def test_runtime_checkable_negative_missing_tool_descriptions(self):
        assert not isinstance(_MissingToolDescriptions(), Agent)
