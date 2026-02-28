from __future__ import annotations

import datetime
import json
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Select, TabbedContent, TabPane

from ._app import (
    _FLYTE_PURPLE,
    _FLYTE_PURPLE_DARK,
    _FLYTE_PURPLE_LIGHT,
    ActionTreeWidget,
    DetailPanel,
)
from ._tracker import ActionTracker

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


def _fmt_attempts(max_attempts_used: int) -> str:
    return f"x{max_attempts_used}"


_SORTABLE_COLUMNS = [
    ("start_time", "Start Time"),
    ("run_name", "Run Name"),
    ("task_name", "Task"),
    ("status", "Status"),
    ("duration", "Duration"),
]


def _compute_runs_table_column_widths(total_width: int) -> list[int]:
    # Account for table borders / separators and keep a sensible floor.
    available = max(total_width - 8, 80)
    min_widths = [12, 12, 10, 10, 19, 12, 16]  # run, task, status, duration, start, attempts, error
    weights = [16, 18, 10, 10, 18, 12, 16]

    base = [max((available * w) // 100, minimum) for w, minimum in zip(weights, min_widths)]
    width_sum = sum(base)
    if width_sum < available:
        extra = available - width_sum
        # Prefer giving extra space to task/run/error columns.
        growth_order = [1, 0, 6, 4, 5, 2, 3]
        i = 0
        while extra > 0:
            target = growth_order[i % len(growth_order)]
            base[target] += 1
            extra -= 1
            i += 1
    return base


class RunsTable(DataTable):
    """Table of all persisted runs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sort_key: str = "start_time"
        self.sort_ascending: bool = False
        self.filter_status: str | None = None
        self.filter_task_name: str | None = None

    def _header_label(self, db_col: str, display: str) -> str:
        if db_col == self.sort_key:
            arrow = " ▲" if self.sort_ascending else " ▼"
            return display + arrow
        return display

    def populate(self) -> None:
        from flyte._persistence._run_store import RunStore

        self.clear(columns=True)
        run_w, task_w, status_w, duration_w, start_w, attempts_w, error_w = _compute_runs_table_column_widths(
            self.size.width
        )
        self.add_column(self._header_label("run_name", "Run Name"), width=run_w)
        self.add_column(self._header_label("task_name", "Task"), width=task_w)
        self.add_column(self._header_label("status", "Status"), width=status_w)
        self.add_column(self._header_label("duration", "Duration"), width=duration_w)
        self.add_column(self._header_label("start_time", "Start Time"), width=start_w)
        self.add_column("Attempts", width=attempts_w)
        self.add_column("Error", width=error_w)
        runs = RunStore.list_runs_sync(
            order_by=self.sort_key,
            ascending=self.sort_ascending,
            status=self.filter_status,
            task_name=self.filter_task_name,
        )
        for r in runs:
            status_text = Text(r.status, style=_STATUS_COLORS.get(r.status, ""))
            error_text = (r.error or "")[:60]
            self.add_row(
                r.run_name,
                r.task_name or "",
                status_text,
                _fmt_duration(r.start_time, r.end_time),
                _fmt_time(r.start_time),
                _fmt_attempts(r.max_attempts_used),
                error_text,
                key=r.run_name,
            )


# Map column index in the DataTable to the db sort key.
_COL_INDEX_TO_SORT_KEY = {
    0: "run_name",
    1: "task_name",
    2: "status",
    3: "duration",
    4: "start_time",
}


class ExploreScreen(Screen):
    """First screen: list of all past runs."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit_app", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "view_run", "View Run"),
        Binding("d", "delete_run", "Delete Run"),
        Binding("c", "clear_all", "Clear All"),
        Binding("s", "cycle_sort", "Sort"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="filter-bar"):
            yield Label("Status:")
            yield Select(
                [("All", "all"), ("Running", "running"), ("Succeeded", "succeeded"), ("Failed", "failed")],
                value="all",
                id="status-filter",
                allow_blank=False,
            )
            yield Label("Task:")
            yield Input(placeholder="Filter by task name...", id="task-filter")
        yield RunsTable(id="runs-table")
        yield Footer()

    def on_mount(self) -> None:
        from flyte._persistence._run_store import RunStore

        RunStore.initialize_sync()
        table = self.query_one("#runs-table", RunsTable)
        table.cursor_type = "row"
        table.populate()

    def _repopulate(self) -> None:
        table = self.query_one("#runs-table", RunsTable)
        table.populate()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "status-filter":
            table = self.query_one("#runs-table", RunsTable)
            table.filter_status = None if event.value == "all" else str(event.value)
            self._repopulate()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "task-filter":
            table = self.query_one("#runs-table", RunsTable)
            table.filter_task_name = event.value.strip() or None
            self._repopulate()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_refresh(self) -> None:
        self._repopulate()

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
        self._repopulate()

    def action_clear_all(self) -> None:
        from flyte._persistence._run_store import RunStore

        RunStore.clear_sync()
        self._repopulate()

    def _toggle_sort(self, sort_key: str) -> None:
        table = self.query_one("#runs-table", RunsTable)
        if table.sort_key == sort_key:
            table.sort_ascending = not table.sort_ascending
        else:
            table.sort_key = sort_key
            table.sort_ascending = False
        self._repopulate()

    def action_cycle_sort(self) -> None:
        table = self.query_one("#runs-table", RunsTable)
        keys = [k for k, _ in _SORTABLE_COLUMNS]
        idx = keys.index(table.sort_key) if table.sort_key in keys else 0
        next_key = keys[(idx + 1) % len(keys)]
        self._toggle_sort(next_key)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        sort_key = _COL_INDEX_TO_SORT_KEY.get(event.column_index)
        if sort_key:
            self._toggle_sort(sort_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_name = str(event.row_key.value)
        self.app.push_screen(RunDetailScreen(run_name))

    def on_resize(self, event: Resize) -> None:
        self._repopulate()


class RunDetailScreen(Screen):
    """Detail screen for a single run, reconstructing the ActionTracker from DB records."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
        Binding("d", "show_details", "Details"),
        Binding("[", "previous_attempt", "Prev Attempt"),
        Binding("]", "next_attempt", "Next Attempt"),
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
            attempts = json.loads(r.attempts_json) if r.attempts_json else None

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
                attempt_count=r.attempt_count,
                attempts=attempts,
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

    def action_previous_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.select_previous_attempt()

    def action_next_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.select_next_attempt()


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
        width: 16;
    }}
    #task-filter {{
        width: 1fr;
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
