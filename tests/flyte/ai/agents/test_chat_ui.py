"""Tests for flyte.ai.agents.chat_ui."""

from unittest.mock import MagicMock

import pytest

from flyte.ai.agents import AgentChatAppEnvironment, AgentResult, CustomTheme
from flyte.ai.agents.chat_ui import _hex_to_rgb, _rgba
from flyte.ai.agents.protocol import Agent


class _StubAgent:
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        return AgentResult(summary=f"echo:{message}")

    def tool_descriptions(self) -> list[dict[str, str]]:
        return [{"name": "n", "signature": "n()", "description": "d"}]


class TestHexHelpers:
    def test_hex_to_rgb_six_char(self):
        assert _hex_to_rgb("#aABBcc") == (170, 187, 204)

    def test_hex_to_rgb_three_char(self):
        assert _hex_to_rgb("#abc") == (170, 187, 204)

    def test_rgba_format(self):
        s = _rgba("#000000", 0.5)
        assert s == "rgba(0, 0, 0, 0.5)"


class TestCustomTheme:
    def test_valid_defaults(self):
        t = CustomTheme()
        css = t.to_css()
        assert "#6F2AEF" in css
        assert "rgba(" in css
        assert ".input-bar button" in css

    def test_custom_colors_in_css(self):
        t = CustomTheme(accent_color="#112233", accent_hover_color="#445566", button_text_color="#ffffff")
        css = t.to_css()
        assert "#112233" in css
        assert "#445566" in css
        assert "#ffffff" in css

    @pytest.mark.parametrize(
        "bad",
        ["red", "#gg0000", "#12", "#12345", ""],
    )
    def test_invalid_hex_raises(self, bad: str):
        with pytest.raises(ValueError, match="CSS hex"):
            CustomTheme(accent_color=bad)


class TestAgentChatAppEnvironment:
    def test_requires_agent(self):
        with pytest.raises(ValueError, match="'agent' is required"):
            AgentChatAppEnvironment(name="test-app", image="auto", agent=None)  # type: ignore[arg-type]

    def test_rejects_non_agent(self):
        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="Agent protocol"):
            AgentChatAppEnvironment(name="test-app", image="auto", agent=NotAnAgent())

    def test_accepts_protocol_agent(self):
        env = AgentChatAppEnvironment(name="test-app", image="auto", agent=_StubAgent())
        assert env.agent is not None
        assert isinstance(env.agent, Agent)

    def test_container_command_empty(self):
        env = AgentChatAppEnvironment(name="test-app", image="auto", agent=_StubAgent())
        assert env.container_command(MagicMock()) == []
