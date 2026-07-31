"""Shared Flyte TUI theme constants and CSS fragments.

Single source of truth for the brand palette and the CSS blocks reused across
the local run TUI (``_app``), the local explore TUI (``_explore``), and the
remote cluster TUI (``_remote``). Each app composes its full stylesheet from
these fragments plus its own app-specific rules.
"""

from __future__ import annotations

# Flyte brand purple palette
_FLYTE_PURPLE = "#7652a2"
_FLYTE_PURPLE_LIGHT = "#f7f5fd"
_FLYTE_PURPLE_DARK = "#171020"
_FLYTE_BORDER = "#DEDDE4"

# Screen / Header / Footer chrome shared by every TUI app.
BASE_CSS = f"""
Screen {{
    background: {_FLYTE_PURPLE_DARK};
}}
Header {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
Footer {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
"""

# Action tree + tabbed detail panel shared by the run/explore/remote screens.
TREE_DETAIL_CSS = f"""
ActionTreeWidget {{
    width: 1fr;
    min-width: 30;
    border: solid {_FLYTE_PURPLE};
    border-title-color: {_FLYTE_PURPLE_LIGHT};
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
#right-tabs {{
    width: 2fr;
}}
DetailPanel {{
    background: {_FLYTE_PURPLE_DARK};
}}
TabPane {{
    padding: 0;
}}
Tabs {{
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
Tab {{
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
Tab.-active {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
Underline {{
    color: {_FLYTE_PURPLE};
}}
_DetailBox {{
    border: solid {_FLYTE_PURPLE};
    border-title-color: {_FLYTE_PURPLE_LIGHT};
    padding: 0 1;
    margin-bottom: 1;
    height: auto;
    color: {_FLYTE_PURPLE_LIGHT};
}}
"""

# Log viewer pane shared by the run and remote run-detail screens.
LOG_VIEWER_CSS = f"""
#log-viewer {{
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
"""

# Interactive condition input panel shared by the run and remote TUIs.
CONDITION_INPUT_CSS = f"""
ConditionInputPanel .condition-prompt,
ConditionInputPanel .condition-description {{
    color: {_FLYTE_PURPLE_LIGHT};
}}
ConditionInputPanel .condition-buttons Button {{
    margin-right: 1;
    color: #ffffff;
    text-style: bold;
}}
ConditionInputPanel .condition-buttons Button:focus {{
    text-style: bold;
}}
ConditionInputPanel .condition-prompt-scroll {{
    max-height: 8;
    height: auto;
    margin-bottom: 1;
}}
ConditionInputPanel .condition-input-row {{
    height: 1;
    min-height: 1;
    layout: horizontal;
    dock: bottom;
    margin-top: 1;
}}
ConditionInputPanel .condition-buttons {{
    height: 1;
    min-height: 1;
    layout: horizontal;
    dock: bottom;
    margin-top: 1;
}}
ConditionInputPanel Markdown.condition-prompt MarkdownHeader {{
    margin: 0 0 1 0;
}}
ConditionInputPanel Markdown.condition-prompt MarkdownParagraph {{
    margin: 0;
}}
ConditionInputPanel Markdown.condition-prompt MarkdownBulletList {{
    margin: 0;
}}
ConditionInputPanel Input {{
    width: 1fr;
    margin-right: 1;
    height: 1;
}}
ConditionInputPanel Input.-textual-compact {{
    background: #1a0a2e;
    color: #ffffff;
}}
ConditionInputPanel Input.-textual-compact:focus {{
    background: #2a1040;
    background-tint: #ffffff 10%;
}}
ConditionInputPanel .condition-string-entry {{
    height: auto;
    dock: bottom;
    margin-top: 1;
}}
ConditionInputPanel .condition-string-entry TextArea {{
    height: 4;
    min-height: 3;
    max-height: 8;
    margin-bottom: 1;
}}
ConditionInputPanel .condition-string-entry TextArea.-textual-compact {{
    background: #1a0a2e;
    color: #ffffff;
}}
ConditionInputPanel .condition-string-entry TextArea.-textual-compact:focus {{
    background: #2a1040;
    background-tint: #ffffff 10%;
}}
ConditionInputPanel .condition-validation-error {{
    color: red;
}}
"""
