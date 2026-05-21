"""Textual screens for the remote TUI."""

from __future__ import annotations

import datetime
from typing import ClassVar

from flyte.cli._tui._app import ActionTreeWidget, DetailPanel
from flyte.cli._tui._explore import StatusSelect, _fmt_duration, _fmt_time
from flyte.cli._tui._tracker import ActionTracker
from flyte.models import ActionPhase
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from ._client import (
    abort_run,
    fetch_log_tail,
    get_run,
    list_actions_for_run,
    list_apps,
    list_runs,
    list_tasks,
    list_triggers,
)
from ._sync import load_run_into_tracker

_STATUS_COLORS = {
    "running": "dodger_blue1",
    "queued": "dodger_blue1",
    "waiting_for_resources": "dodger_blue1",
    "initializing": "dodger_blue1",
    "succeeded": "green",
    "failed": "red",
    "aborted": "red",
    "timed_out": "red",
}

_PHASE_FILTER_MAP = {
    "all": None,
    "running": (
        ActionPhase.RUNNING,
        ActionPhase.QUEUED,
        ActionPhase.WAITING_FOR_RESOURCES,
        ActionPhase.INITIALIZING,
    ),
    "succeeded": (ActionPhase.SUCCEEDED,),
    "failed": (
        ActionPhase.FAILED,
        ActionPhase.ABORTED,
        ActionPhase.TIMED_OUT,
    ),
}


def _run_duration(run) -> str:
    action = run.action
    start = action.start_time
    if action.done():
        end_pb = action.pb2.status
        if end_pb.HasField("end_time"):
            end = end_pb.end_time.ToDatetime()
            return _fmt_duration(start.timestamp(), end.timestamp())
    return _fmt_duration(start.timestamp(), None)


class EntityTable(DataTable):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down,j", "cursor_down", "Cursor Down", show=False),
        Binding("up,k", "cursor_up", "Cursor Up", show=False),
    ]


class RunsScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "View Run"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="filter-bar"):
            yield Label("Status:")
            yield StatusSelect(
                [
                    ("All", "all"),
                    ("Running", "running"),
                    ("Succeeded", "succeeded"),
                    ("Failed", "failed"),
                ],
                value="all",
                id="status-filter",
                allow_blank=False,
            )
            yield Label("Task:")
            yield Input(placeholder="Filter by task name...", id="run-task-filter")
        yield EntityTable(id="runs-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", EntityTable)
        table.cursor_type = "row"
        table.focus()
        self._repopulate()

    def _repopulate(self) -> None:
        table = self.query_one("#runs-table", EntityTable)
        status_sel = self.query_one("#status-filter", Select)
        task_inp = self.query_one("#run-task-filter", Input)
        phase_key = str(status_sel.value or "all")
        in_phase = _PHASE_FILTER_MAP.get(phase_key)
        task_filter = task_inp.value.strip() or None

        table.clear(columns=True)
        w = max(table.size.width - 8, 80)
        table.add_column("Run Name", width=int(w * 0.28))
        table.add_column("Task", width=int(w * 0.22))
        table.add_column("Status", width=int(w * 0.14))
        table.add_column("Duration", width=int(w * 0.14))
        table.add_column("Started", width=int(w * 0.22))

        try:
            runs = list_runs(limit=200, task_name=task_filter, in_phase=in_phase)
        except Exception as exc:
            table.add_row(f"Error: {exc}", "", "", "", "")
            return

        for run in runs:
            phase = str(run.phase).lower() if hasattr(run.phase, "value") else str(run.phase)
            status_text = Text(phase, style=_STATUS_COLORS.get(phase, ""))
            task = run.action.task_name or ""
            st = run.action.start_time
            started = _fmt_time(st.timestamp() if hasattr(st, "timestamp") else float(st))
            table.add_row(
                run.name,
                task,
                status_text,
                _run_duration(run),
                started,
                key=run.name,
            )

    def action_refresh(self) -> None:
        self._repopulate()

    def action_open_detail(self) -> None:
        table = self.query_one("#runs-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        run_name = str(row_key.value)
        self.app.push_screen(RunDetailScreen(run_name))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "status-filter":
            self._repopulate()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "run-task-filter":
            self._repopulate()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(RunDetailScreen(str(event.row_key.value)))

    def on_resize(self, event: Resize) -> None:
        self._repopulate()


class TasksScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "View Task"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="filter-bar"):
            yield Label("Name:")
            yield Input(placeholder="Filter by task name...", id="task-filter")
        yield EntityTable(id="tasks-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tasks-table", EntityTable).cursor_type = "row"
        self._repopulate()

    def _repopulate(self) -> None:
        table = self.query_one("#tasks-table", EntityTable)
        name_filter = self.query_one("#task-filter", Input).value.strip() or None
        table.clear(columns=True)
        table.add_column("Name", width=30)
        table.add_column("Version", width=16)
        table.add_column("Type", width=20)
        table.add_column("Env", width=20)
        try:
            tasks = list_tasks(limit=200, task_name=name_filter)
        except Exception as exc:
            table.add_row(f"Error: {exc}", "", "", "")
            return
        for t in tasks:
            pb = t.pb2
            env = ""
            if pb.metadata.HasField("environment_name"):
                env = pb.metadata.environment_name
            table.add_row(
                t.name,
                pb.id.version,
                pb.task_template.type if pb.HasField("task_template") else "",
                env,
                key=t.name,
            )

    def action_refresh(self) -> None:
        self._repopulate()

    def action_open_detail(self) -> None:
        table = self.query_one("#tasks-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self.app.push_screen(EntityDetailScreen("Task", str(row_key.value)))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "task-filter":
            self._repopulate()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(EntityDetailScreen("Task", str(event.row_key.value)))


class AppsScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "View App"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield EntityTable(id="apps-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#apps-table", EntityTable).cursor_type = "row"
        self._repopulate()

    def _repopulate(self) -> None:
        table = self.query_one("#apps-table", EntityTable)
        table.clear(columns=True)
        table.add_column("Name", width=24)
        table.add_column("Status", width=18)
        table.add_column("Endpoint", width=40)
        try:
            apps = list_apps(limit=200)
        except Exception as exc:
            table.add_row(f"Error: {exc}", "", "")
            return
        for app in apps:
            status = app.deployment_status.name.replace("DEPLOYMENT_STATUS_", "").lower()
            endpoint = app.endpoint or ""
            table.add_row(app.name, status, endpoint, key=app.name)

    def action_refresh(self) -> None:
        self._repopulate()

    def action_open_detail(self) -> None:
        table = self.query_one("#apps-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self.app.push_screen(EntityDetailScreen("App", str(row_key.value)))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(EntityDetailScreen("App", str(event.row_key.value)))


class TriggersScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "View Trigger"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield EntityTable(id="triggers-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#triggers-table", EntityTable).cursor_type = "row"
        self._repopulate()

    def _repopulate(self) -> None:
        table = self.query_one("#triggers-table", EntityTable)
        table.clear(columns=True)
        table.add_column("Name", width=24)
        table.add_column("Task", width=24)
        table.add_column("Active", width=10)
        try:
            triggers = list_triggers(limit=200)
        except Exception as exc:
            table.add_row(f"Error: {exc}", "", "")
            return
        for tr in triggers:
            tname = tr.pb2.id.name.task_name
            name = tr.pb2.id.name.name
            active = "yes" if tr.pb2.spec.active else "no"
            table.add_row(name, tname, active, key=f"{tname}/{name}")

    def action_refresh(self) -> None:
        self._repopulate()

    def action_open_detail(self) -> None:
        table = self.query_one("#triggers-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        key = str(row_key.value)
        if "/" in key:
            task_name, name = key.split("/", 1)
            self.app.push_screen(EntityDetailScreen("Trigger", name, task_name=task_name))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = str(event.row_key.value)
        if "/" in key:
            task_name, name = key.split("/", 1)
            self.app.push_screen(EntityDetailScreen("Trigger", name, task_name=task_name))


class EntityDetailScreen(Screen):
    """Generic read-only detail for Task / App / Trigger."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
    ]

    def __init__(self, kind: str, name: str, *, task_name: str | None = None) -> None:
        super().__init__()
        self._kind = kind
        self._name = name
        self._task_name = task_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(id="detail-body"), id="detail-scroll")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"{self._kind}: {self._name}"
        body = self.query_one("#detail-body", Static)
        lines: list[str] = []
        try:
            if self._kind == "Task":
                import flyte.remote as remote

                t = remote.Task.get(name=self._name, version="latest")
                details = t.fetch()
                lines.append(f"name:       {self._name}")
                lines.append(f"version:    {details.pb2.id.version}")
                lines.append(f"type:       {details.pb2.task_template.type}")
                lines.append(f"deployed:   {details.pb2.metadata.deployed_at.ToDatetime()}")
            elif self._kind == "App":
                import flyte.remote as remote

                apps = {a.name: a for a in remote.App.listall(limit=500)}
                app = apps.get(self._name)
                if app is None:
                    lines.append(f"App '{self._name}' not found.")
                else:
                    lines.append(f"name:       {app.name}")
                    lines.append(f"status:     {app.deployment_status.name}")
                    lines.append(f"endpoint:   {app.endpoint or '(none)'}")
                    lines.append(f"url:        {app.url}")
            elif self._kind == "Trigger":
                import flyte.remote as remote

                assert self._task_name
                tr = remote.Trigger.get(name=self._name, task_name=self._task_name)
                lines.append(f"name:       {tr.name}")
                lines.append(f"task:       {tr.task_name}")
                lines.append(f"active:     {tr.is_active}")
        except Exception as exc:
            lines.append(f"Error loading {self._kind}: {exc}")
        body.update("\n".join(lines))

    def action_go_back(self) -> None:
        self.app.pop_screen()


class _LogViewer(RichLog):
    def write_line(self, text: str) -> None:
        self.write(text, scroll_end=True)


class RunDetailScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "show_details", "Details"),
        Binding("l", "show_logs", "Logs"),
        Binding("[", "previous_attempt", "Prev Attempt"),
        Binding("]", "next_attempt", "Next Attempt"),
        Binding("a", "abort_run", "Abort"),
    ]

    def __init__(self, run_name: str) -> None:
        super().__init__()
        self._run_name = run_name
        self._tracker = ActionTracker()
        self._last_version = -1
        self._selected_action: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            tree = ActionTreeWidget(self._tracker, id="action-tree")
            tree.border_title = f"Run: {self._run_name}"
            yield tree
            with TabbedContent(initial="tab-details", id="right-tabs"):
                with TabPane("Details", id="tab-details"):
                    yield DetailPanel(self._tracker, id="detail-panel")
                with TabPane("Logs", id="tab-logs"):
                    yield _LogViewer(
                        id="log-viewer",
                        auto_scroll=True,
                        wrap=True,
                        markup=False,
                        highlight=False,
                        max_lines=10_000,
                    )
                with TabPane("Run Info", id="tab-run-info"):
                    yield VerticalScroll(Static(id="run-info-body"), id="detail-scroll")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Run: {self._run_name}"
        poll = getattr(self.app, "poll_interval", 2.0)
        self.set_interval(poll, self._poll_run)
        self._reload_run_data()
        self._refresh_logs()

    @work(thread=True)
    def _reload_run_data(self) -> None:
        try:
            actions = list_actions_for_run(self._run_name)
            load_run_into_tracker(self._tracker, actions, fetch_io=True)
            run = get_run(self._run_name)
            info_lines = [
                f"run:        {self._run_name}",
                f"phase:      {run.phase}",
                f"task:       {run.action.task_name or 'n/a'}",
                f"url:        {run.url}",
                f"actions:    {len(actions)}",
                f"updated:    {datetime.datetime.now().strftime('%H:%M:%S')}",
            ]
            phase = str(run.phase)
            self.app.call_from_thread(self._apply_run_data, info_lines, phase)
        except Exception as exc:
            self.app.call_from_thread(self._apply_run_error, str(exc))

    def _apply_run_data(self, info_lines: list[str], phase: str) -> None:
        self._last_version = self._tracker.version
        try:
            self.query_one("#run-info-body", Static).update("\n".join(info_lines))
            self.query_one("#action-tree", ActionTreeWidget).refresh_from_tracker()
            self.query_one("#detail-panel", DetailPanel).refresh_detail()
            self.sub_title = phase
        except Exception:
            pass

    def _apply_run_error(self, message: str) -> None:
        self.sub_title = f"error: {message}"

    def _refresh_run(self) -> None:
        self._reload_run_data()

    def _poll_run(self) -> None:
        try:
            run = get_run(self._run_name)
            if not run.done():
                self._reload_run_data()
        except Exception:
            pass

    @work(thread=True, exit_on_error=False)
    def _refresh_logs(self) -> None:
        action_name = self._selected_action
        try:
            lines = fetch_log_tail(
                self._run_name,
                action_name,
                max_lines=300,
                show_ts=True,
            )
            text = "\n".join(lines) if lines else "(no logs yet)"
            self.app.call_from_thread(self._set_logs, text)
        except Exception as exc:
            self.app.call_from_thread(self._set_logs, f"[log error] {exc}")

    def _set_logs(self, text: str) -> None:
        try:
            viewer = self.query_one("#log-viewer", _LogViewer)
            viewer.clear()
            viewer.write(text)
        except Exception:
            pass

    def on_tree_node_selected(self, event) -> None:
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.action_id = event.node.data
        self._selected_action = str(event.node.data) if event.node.data else None
        self._refresh_logs()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._reload_run_data()
        self._refresh_logs()

    def action_show_details(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"

    def action_show_logs(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-logs"
        self._refresh_logs()

    def action_previous_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        self.query_one("#detail-panel", DetailPanel).select_previous_attempt()

    def action_next_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        self.query_one("#detail-panel", DetailPanel).select_next_attempt()

    def action_abort_run(self) -> None:
        try:
            abort_run(self._run_name)
            self.notify("Run abort requested")
            self._refresh_run()
        except Exception as exc:
            self.notify(f"Abort failed: {exc}", severity="error")
