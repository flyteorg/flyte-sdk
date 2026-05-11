"""flyte.ai.chat — FastAPI chat UI and HTML/CSS assets for Flyte agents."""

from ._css import CUSTOM_THEME_CSS_TEMPLATE, DEFAULT_CSS
from ._html import CHAT_HTML_TEMPLATE, build_chat_html
from .app import AgentChatAppEnvironment, CustomTheme

__all__ = [
    "CHAT_HTML_TEMPLATE",
    "CUSTOM_THEME_CSS_TEMPLATE",
    "DEFAULT_CSS",
    "AgentChatAppEnvironment",
    "CustomTheme",
    "build_chat_html",
]
