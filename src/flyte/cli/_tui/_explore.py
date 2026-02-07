from __future__ import annotations

import datetime
import json
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane
from textual.widgets.tree import TreeNode

from ._app import (
    ActionTreeWidget,
    DetailPanel,
    _FLYTE_BORDER,
    _FLYTE_PURPLE,
    _FLYTE_PURPLE_DARK,
    _FLYTE_PURPLE_LIGHT,
    _DetailBox,
)
from ._tracker import ActionStatus, ActionTracker

_STATUS_COLORS = {
    "running": "dodger_blue1",
    "succeeded": "green",
    "failed": "red",
}


def _fmt_duration(start: float | None, end: float | None) -> str:
    if start is None:
        return ""
    if end is None:
        return "running..."
    return f"{end - start:.2f}s"


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class RunsTable(DataTable):
    """Table of all persisted runs."""

    def populate(self) -> None:
        from flyte._persistence._run_store import RunStore

        self.clear(columns=True)
        self.add_columns("Run Name", "Task", "Status", "Duration", "Start Time", "Error")
        runs = RunStore.list_runs_sync()
        for r in runs:
            status_text = Text(r.status, style=_STATUS_COLORS.get(r.status, ""))
            error_text = (r.error or "")[:60]
            self.add_row(
                r.run_name,
                r.task_name or "",
                status_text,
                _fmt_duration(r.start_time, r.end_time),
                _fmt_time(r.start_time),
                error_text,
                key=r.run_name,
            )


class ExploreScreen(Screen):
    """First screen: list of all past runs."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit_app", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "view_run", "View Run"),
        Binding("d", "delete_run", "Delete Run"),
        Binding("c", "clear_all", "Clear All"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RunsTable(id="runs-table")
        yield Footer()

    def on_mount(self) -> None:
        from flyte._persistence._run_store import RunStore

        RunStore.initialize_sync()
        table = self.query_one("#runs-table", RunsTable)
        table.cursor_type = "row"
        table.populate()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_refresh(self) -> None:
        table = self.query_one("#runs-table", RunsTable)
        table.populate()

    def action_view_run(self) -> None:
        table = self.query_one("#runs-table", RunsTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        run_name = str(row_key.value)
        self.app.push_screen(RunDetailScreen(run_name))

    def action_delete_run(self) -> None:
        from flyte._persistence._run_store import RunStore

        table = self.query_one("#runs-table", RunsTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        run_name = str(row_key.value)
        RunStore.delete_run_sync(run_name)
        table.populate()

    def action_clear_all(self) -> None:
        from flyte._persistence._run_store import RunStore

        RunStore.clear_sync()
        table = self.query_one("#runs-table", RunsTable)
        table.populate()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_name = str(event.row_key.value)
        self.app.push_screen(RunDetailScreen(run_name))


class RunDetailScreen(Screen):
    """Detail screen for a single run, reconstructing the ActionTracker from DB records."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
        Binding("d", "show_details", "Details"),
    ]

    def __init__(self, run_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._run_name = run_name
        self._tracker = ActionTracker()
        self._rebuild_tracker()

    def _rebuild_tracker(self) -> None:
        from flyte._persistence._run_store import RunStore

        records = RunStore.list_actions_for_run_sync(self._run_name)
        for r in records:
            # Deserialize JSON fields stored in the DB
            inputs = json.loads(r.inputs) if r.inputs else None
            context = json.loads(r.context) if r.context else None
            log_links = json.loads(r.log_links) if r.log_links else None

            self._tracker.record_start(
                action_id=r.action_name,
                task_name=r.task_name or r.action_name,
                parent_id=r.parent_id,
                short_name=r.short_name,
                inputs=inputs,
                output_path=r.output_path,
                has_report=r.has_report,
                cache_enabled=r.cache_enabled,
                cache_hit=r.cache_hit,
                context=context,
                group=r.group_name,
                log_links=log_links,
            )
            if r.status == "succeeded":
                self._tracker.record_complete(action_id=r.action_name, outputs=r.outputs)
            elif r.status == "failed":
                self._tracker.record_failure(action_id=r.action_name, error=r.error or "Unknown error")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            tree = ActionTreeWidget(self._tracker, id="action-tree")
            tree.border_title = f"Run: {self._run_name}"
            yield tree
            with TabbedContent(initial="tab-details", id="right-tabs"):
                with TabPane("Details", id="tab-details"):
                    yield DetailPanel(self._tracker, id="detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Run: {self._run_name}"
        tree = self.query_one("#action-tree", ActionTreeWidget)
        tree.refresh_from_tracker()

    def on_tree_node_selected(self, event: Any) -> None:
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.action_id = event.node.data

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_show_details(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"


class ExploreTUIApp(App[None]):
    """Standalone TUI app for browsing past local runs."""

    CSS = f"""
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
    Horizontal {{
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
    RunsTable {{
        height: 1fr;
        background: {_FLYTE_PURPLE_DARK};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    DataTable > .datatable--cursor {{
        background: {_FLYTE_PURPLE};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    """

    def on_mount(self) -> None:
        self.title = "Flyte Explore"
        self.push_screen(ExploreScreen())
