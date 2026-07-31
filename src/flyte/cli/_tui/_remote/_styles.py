"""Stylesheet for the remote cluster TUI.

Composed from the shared theme fragments in ``flyte.cli._tui._theme`` plus the
remote-specific screen rules (projects home, project hub, entity tables).
"""

from __future__ import annotations

from .._theme import (
    _FLYTE_PURPLE,
    _FLYTE_PURPLE_DARK,
    _FLYTE_PURPLE_LIGHT,
    BASE_CSS,
    CONDITION_INPUT_CSS,
    LOG_VIEWER_CSS,
    TREE_DETAIL_CSS,
)

# Rules unique to the remote screens (Projects / ProjectHub / RunDetail).
_REMOTE_CSS = f"""
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
    layout: vertical;
}}
#hub-header {{
    height: 1;
    padding: 0 1;
}}
#hub-header #section-title {{
    width: 1fr;
    color: {_FLYTE_PURPLE_LIGHT};
    text-style: bold;
    height: 1;
}}
#page-indicator {{
    width: auto;
    max-width: 1fr;
    height: 1;
    color: {_FLYTE_PURPLE_LIGHT};
    text-align: right;
    text-style: italic;
}}
ProjectHubScreen #filter-bar {{
    height: 3;
    padding: 0 1;
}}
#hub-table {{
    height: 1fr;
    min-height: 5;
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
ConditionResultPanel {{
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #projects-content {{
    height: 1fr;
    layout: horizontal;
}}
ProjectsScreen #recent-sidebar {{
    width: 24;
    min-width: 18;
    height: 1fr;
    layout: vertical;
    border: solid {_FLYTE_PURPLE};
    border-title-color: {_FLYTE_PURPLE_LIGHT};
    background: {_FLYTE_PURPLE_DARK};
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #recent-projects {{
    height: 1fr;
    padding: 0;
    border: none;
    background: transparent;
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #recent-projects ListItem.-highlight {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #sidebar-footer {{
    height: auto;
    width: 1fr;
    layout: vertical;
    border-top: solid {_FLYTE_PURPLE};
    background: {_FLYTE_PURPLE_DARK};
}}
ProjectsScreen #docs-link {{
    height: 1;
    padding: 0 1;
    color: {_FLYTE_PURPLE_LIGHT};
    text-style: underline;
}}
ProjectsScreen #docs-link:hover {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #docs-link:focus {{
    background: {_FLYTE_PURPLE};
    color: {_FLYTE_PURPLE_LIGHT};
}}
ProjectsScreen #projects-main {{
    width: 1fr;
    height: 1fr;
    layout: vertical;
}}
ProjectsScreen #projects-table {{
    height: 1fr;
    min-height: 8;
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

APP_CSS = BASE_CSS + TREE_DETAIL_CSS + LOG_VIEWER_CSS + CONDITION_INPUT_CSS + _REMOTE_CSS
