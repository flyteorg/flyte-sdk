"""Tests for flyte.ai.agents._html and _css."""

import re

import flyte.ai.agents as agents_pkg
from flyte.ai.agents import (
    DEFAULT_CSS,
    Agent,
    AgentChatAppEnvironment,
    AgentResult,
    CodeModeAgent,
    CustomTheme,
)
from flyte.ai.agents._html import CHAT_HTML_TEMPLATE, _build_action_buttons_html, build_chat_html


class TestDefaultCss:
    def test_non_empty(self):
        assert len(DEFAULT_CSS) > 1000
        assert "body {" in DEFAULT_CSS
        assert ".sidebar" in DEFAULT_CSS


class TestBuildActionButtonsHtml:
    def test_empty(self):
        assert _build_action_buttons_html([]) == ""

    def test_single_button(self):
        html = _build_action_buttons_html([{"button_text": "Docs", "button_url": "https://example.com"}])
        assert "action-btn-group" in html
        assert "has-menu" not in html
        assert "https://example.com" in html
        assert "Docs" in html
        assert "actionChevron" not in html

    def test_multiple_buttons_menu(self):
        html = _build_action_buttons_html(
            [
                {"button_text": "Primary", "button_url": "https://a.test"},
                {"button_text": "Secondary", "button_url": "https://b.test"},
            ]
        )
        assert "has-menu" in html
        assert "actionChevron" in html
        assert "https://a.test" in html
        assert "https://b.test" in html


class TestBuildChatHtml:
    def test_contains_title_and_default_css(self):
        html = build_chat_html(title="My Agent")
        assert "<title>My Agent</title>" in html
        assert DEFAULT_CSS[:80] in html or "body {" in html

    def test_custom_css_block(self):
        html = build_chat_html(title="T", custom_css=".foo { color: red; }")
        assert "Custom overrides" in html
        assert ".foo { color: red; }" in html

    def test_logo_img_when_url(self):
        html = build_chat_html(title="T", logo_url="https://cdn.example/logo.png")
        assert 'class="header-logo"' in html
        assert 'src="https://cdn.example/logo.png"' in html

    def test_no_logo_when_none(self):
        html = build_chat_html(title="T", logo_url=None)
        # Default CSS defines .header-logo; only an explicit logo_url emits <img>.
        assert '<img class="header-logo"' not in html

    def test_subtitle_paragraph(self):
        html = build_chat_html(title="T", subtitle="Sub text here")
        assert 'class="app-description"' in html
        assert "Sub text here" in html

    def test_action_buttons_pass_through(self):
        html = build_chat_html(
            title="T",
            additional_buttons=[{"button_text": "X", "button_url": "https://x"}],
        )
        assert "https://x" in html

    def test_template_placeholders_replaced(self):
        raw = CHAT_HTML_TEMPLATE
        assert "$TITLE" in raw
        out = build_chat_html(title="Replaced")
        assert "$TITLE" not in out
        assert "$CSS" not in out
        assert "$LOGO" not in out
        assert "$SUBTITLE" not in out
        assert "$ACTION_BUTTONS" not in out

    def test_inline_script_ndjson_split_uses_backslash_n_escape(self):
        """Avoid Python newline escapes inside CHAT_HTML_TEMPLATE breaking JS (SyntaxError)."""
        html = build_chat_html(title="T")
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert m is not None
        script_body = m.group(1)
        assert "buffer.split('\\n')" in script_body


class TestPackageExports:
    def test_public_import_surface(self):
        assert set(agents_pkg.__all__) == {
            "DEFAULT_CSS",
            "Agent",
            "AgentChatAppEnvironment",
            "AgentResult",
            "CodeModeAgent",
            "CustomTheme",
        }
        assert AgentChatAppEnvironment.__name__ == "AgentChatAppEnvironment"
        assert CodeModeAgent.__name__ == "CodeModeAgent"
        assert CustomTheme.__name__ == "CustomTheme"
        assert AgentResult.__name__ == "AgentResult"
        assert hasattr(Agent, "run")
