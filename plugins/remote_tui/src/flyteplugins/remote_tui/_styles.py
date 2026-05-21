"""Shared Flyte TUI theme (matches ``flyte.cli._tui``)."""

from __future__ import annotations

_FLYTE_PURPLE = "#7652a2"
_FLYTE_PURPLE_LIGHT = "#f7f5fd"
_FLYTE_PURPLE_DARK = "#171020"

APP_CSS = f"""
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
#project-sidebar {{
    width: 16;
    min-width: 14;
    border: solid {_FLYTE_PURPLE};
    border-title-color: {_FLYTE_PURPLE_LIGHT};
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
    padding: 1 0;
    height: 1fr;
}}
#project-sidebar ListItem.-highlight {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
#hub-content {{
    width: 1fr;
    height: 1fr;
}}
#section-title {{
    padding: 0 1;
    color: {_FLYTE_PURPLE_LIGHT};
    text-style: bold;
    height: 1;
}}
#section-filter {{
    width: 1fr;
}}
ProjectHubScreen Horizontal {{
    height: 1fr;
}}
RunDetailScreen Horizontal {{
    height: 1fr;
}}
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
#log-viewer {{
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
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
#filter-bar {{
    height: 3;
    padding: 0 1;
    background: {_FLYTE_PURPLE_DARK};
}}
#filter-bar Label {{
    padding: 1 1 0 0;
    color: {_FLYTE_PURPLE_LIGHT};
    width: auto;
}}
#status-filter {{
    width: 18;
}}
#task-filter, #run-task-filter {{
    width: 1fr;
}}
EntityTable {{
    height: 1fr;
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
DataTable > .datatable--cursor {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
#detail-scroll {{
    height: 1fr;
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
    padding: 0 1;
}}
#detail-scroll Static {{
    color: {_FLYTE_PURPLE_LIGHT};
}}
"""
