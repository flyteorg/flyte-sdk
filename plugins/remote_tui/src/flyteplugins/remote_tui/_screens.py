"""Textual screens mirroring the Flyte 2 Devbox UI hierarchy."""

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
    ListItem,
    ListView,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from ._client import (
    abort_run,
    activate_project,
    fetch_log_tail,
    get_run,
    list_actions_for_run,
    list_apps,
    list_projects,
    list_runs,
    list_tasks,
    list_triggers,
)
from ._context import as_remote_app, list_scope
from ._sync import load_run_into_tracker


def _format_app_deployment_status(status: int | str) -> str:
    """Human-readable deployment status (matches flyte.remote.App.__rich_repr__)."""
    if isinstance(status, str):
        return status.removeprefix("DEPLOYMENT_STATUS_")
    from flyteidl2.app import app_definition_pb2

    return app_definition_pb2.Status.DeploymentStatus.Name(status)[len("DEPLOYMENT_STATUS_") :]


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

_NAV_RUNS = "nav-runs"
_NAV_TRIGGERS = "nav-triggers"
_NAV_TASKS = "nav-tasks"
_NAV_APPS = "nav-apps"


def _run_duration(run) -> str:
    action = run.action
    start = action.start_time
    if action.done():
        end_pb = action.pb2.status
        if end_pb.HasField("end_time"):
            end = end_pb.end_time.ToDatetime()
            return _fmt_duration(start.timestamp(), end.timestamp())
    return _fmt_duration(start.timestamp(), None)


def _format_labels(project) -> str:
    if project.pb2.labels and project.pb2.labels.values:
        return ", ".join(f"{k}={v}" for k, v in project.pb2.labels.values.items())
    return "None"


def _phase_icon(phase: str) -> str:
    if phase in ("succeeded",):
        return "✓"
    if phase in ("failed", "aborted", "timed_out"):
        return "✗"
    return "●"


class EntityTable(DataTable):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down,j", "cursor_down", "Cursor Down", show=False),
        Binding("up,k", "cursor_up", "Cursor Up", show=False),
    ]


class ProjectsScreen(Screen):
    """Top-level project list (Devbox home)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_project", "Open Project"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="filter-bar"):
            yield Label("Search:")
            yield Input(placeholder="Filter projects...", id="project-search")
        yield EntityTable(id="projects-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Projects"
        table = self.query_one("#projects-table", EntityTable)
        table.cursor_type = "row"
        table.focus()
        self._repopulate()

    def _repopulate(self) -> None:
        table = self.query_one("#projects-table", EntityTable)
        search = self.query_one("#project-search", Input).value.strip().lower()
        table.clear(columns=True)
        table.add_column("Name", width=28)
        table.add_column("Labels", width=24)
        table.add_column("ID", width=28)
        try:
            projects = list_projects()
        except Exception as exc:
            table.add_row(f"Error: {exc}", "", "")
            return
        count = 0
        for proj in projects:
            name = proj.pb2.name or proj.pb2.id
            if search and search not in name.lower() and search not in proj.pb2.id.lower():
                continue
            count += 1
            table.add_row(name, _format_labels(proj), proj.pb2.id, key=proj.pb2.id)
        self.sub_title = f"{count} total"

    def action_refresh(self) -> None:
        self._repopulate()

    def _open_project(self, project_id: str) -> None:
        app = as_remote_app(self.app)
        cluster = app.cluster
        if cluster is None:
            return
        activate_project(
            config=app._config,
            project=project_id,
            domain=cluster.domain,
            org=cluster.org,
        )
        app.selected_project = project_id
        app.set_subtitle_for_project(project_id)
        self.app.push_screen(ProjectHubScreen(project_id))

    def action_open_project(self) -> None:
        table = self.query_one("#projects-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self._open_project(str(row_key.value))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "project-search":
            self._repopulate()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_project(str(event.row_key.value))

    def on_resize(self, event: Resize) -> None:
        self._repopulate()


class ProjectHubScreen(Screen):
    """Project workspace: sidebar (Runs / Triggers / Tasks / Apps) + list."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back_to_projects", "Projects"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "Open"),
        Binding("1", "show_runs", "Runs", show=False),
        Binding("2", "show_triggers", "Triggers", show=False),
        Binding("3", "show_tasks", "Tasks", show=False),
        Binding("4", "show_apps", "Apps", show=False),
    ]

    def __init__(self, project_name: str) -> None:
        super().__init__()
        self._project = project_name
        self._section = "runs"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(
                ListItem(Static("Runs"), id=_NAV_RUNS),
                ListItem(Static("Triggers"), id=_NAV_TRIGGERS),
                ListItem(Static("Tasks"), id=_NAV_TASKS),
                ListItem(Static("Apps"), id=_NAV_APPS),
                id="project-sidebar",
            )
            with VerticalScroll(id="hub-content"):
                yield Static("", id="section-title")
                with Horizontal(id="filter-bar"):
                    yield Label("Status:", id="status-label")
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
                    yield Label("Filter:", id="filter-label")
                    yield Input(placeholder="", id="section-filter")
                yield EntityTable(id="hub-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self._project
        self.query_one("#hub-table", EntityTable).cursor_type = "row"
        sidebar = self.query_one("#project-sidebar", ListView)
        sidebar.border_title = self._project
        self._select_section("runs")
        self.query_one("#hub-table", EntityTable).focus()

    def _select_section(self, section: str) -> None:
        self._section = section
        titles = {
            "runs": "Runs",
            "triggers": "Triggers",
            "tasks": "Tasks",
            "apps": "Apps",
        }
        self.query_one("#section-title", Static).update(titles[section])
        status_label = self.query_one("#status-label", Label)
        filter_label = self.query_one("#filter-label", Label)
        status_filter = self.query_one("#status-filter", Select)
        section_filter = self.query_one("#section-filter", Input)
        if section == "runs":
            status_label.display = True
            status_filter.display = True
            filter_label.update("Task:")
            section_filter.placeholder = "Filter by task name..."
        elif section == "tasks":
            status_label.display = False
            status_filter.display = False
            filter_label.update("Name:")
            section_filter.placeholder = "Filter by task name..."
        else:
            status_label.display = False
            status_filter.display = False
            filter_label.display = False
            section_filter.display = section == "triggers"
            if section == "triggers":
                filter_label.display = True
                filter_label.update("Search:")
                section_filter.placeholder = "Filter triggers..."
            else:
                section_filter.display = False
        self._repopulate()

    def _repopulate(self) -> None:
        scope = list_scope(as_remote_app(self.app))
        table = self.query_one("#hub-table", EntityTable)
        table.clear(columns=True)
        filt = self.query_one("#section-filter", Input).value.strip() or None

        if self._section == "runs":
            self._populate_runs(table, scope, filt)
        elif self._section == "triggers":
            self._populate_triggers(table, filt)
        elif self._section == "tasks":
            self._populate_tasks(table, scope, filt)
        elif self._section == "apps":
            self._populate_apps(table)

    def _populate_runs(self, table: EntityTable, scope: dict[str, str], task_filter: str | None) -> None:
        status_sel = self.query_one("#status-filter", Select)
        phase_key = str(status_sel.value or "all")
        in_phase = _PHASE_FILTER_MAP.get(phase_key)
        table.add_column("", width=3)
        table.add_column("Run", width=22)
        table.add_column("Trigger", width=18)
        table.add_column("Duration", width=10)
        table.add_column("Started", width=14)
        table.add_column("Ended", width=14)
        try:
            runs = list_runs(
                limit=200,
                task_name=task_filter,
                in_phase=in_phase,
                **scope,
            )
        except Exception as exc:
            table.add_row("", f"Error: {exc!s}", "", "", "", "")
            return
        self.sub_title = f"{len(runs)} total"
        for run in runs:
            phase = str(run.phase).lower() if hasattr(run.phase, "value") else str(run.phase)
            icon = Text(_phase_icon(phase), style=_STATUS_COLORS.get(phase, ""))
            task = run.action.task_name or "-"
            st = run.action.start_time
            started = _fmt_time(st.timestamp() if hasattr(st, "timestamp") else float(st))
            ended = ""
            if run.action.done() and run.action.pb2.status.HasField("end_time"):
                end = run.action.pb2.status.end_time.ToDatetime()
                ended = _fmt_time(end.timestamp())
            table.add_row(
                icon,
                run.name,
                task,
                _run_duration(run),
                started,
                ended,
                key=run.name,
            )

    def _populate_triggers(self, table: EntityTable, search: str | None) -> None:
        table.add_column("Name", width=24)
        table.add_column("Task", width=24)
        table.add_column("Active", width=10)
        try:
            triggers = list_triggers(limit=200)
        except Exception as exc:
            table.add_row(f"Error: {exc!s}", "", "")
            return
        rows = []
        for tr in triggers:
            tname = tr.task_name
            name = tr.name
            if search and search.lower() not in name.lower() and search.lower() not in tname.lower():
                continue
            active = "yes" if tr.is_active else "no"
            rows.append((name, tname, active, f"{tname}/{name}"))
        self.sub_title = f"{len(rows)} total"
        for name, tname, active, key in rows:
            table.add_row(name, tname, active, key=key)

    def _populate_tasks(self, table: EntityTable, scope: dict[str, str], name_filter: str | None) -> None:
        table.add_column("Name", width=28)
        table.add_column("Version", width=14)
        table.add_column("Short name", width=18)
        table.add_column("Env", width=18)
        try:
            tasks = list_tasks(limit=200, task_name=name_filter, **scope)
        except Exception as exc:
            table.add_row(f"Error: {exc!s}", "", "", "")
            return
        self.sub_title = f"{len(tasks)} total"
        for t in tasks:
            meta = t.pb2.metadata
            # Proto3 string fields have no presence; read values directly (empty string if unset).
            short = meta.short_name or "-"
            env = meta.environment_name or "-"
            table.add_row(
                t.name,
                t.version,
                short,
                env,
                key=t.name,
            )

    def _populate_apps(self, table: EntityTable) -> None:
        table.add_column("Name", width=24)
        table.add_column("Status", width=18)
        table.add_column("Endpoint", width=36)
        try:
            apps = list_apps(limit=200)
        except Exception as exc:
            table.add_row(f"Error: {exc!s}", "", "")
            return
        self.sub_title = f"{len(apps)} total"
        for app in apps:
            status = _format_app_deployment_status(app.deployment_status).lower()
            table.add_row(app.name, status, app.endpoint or "", key=app.name)

    def action_refresh(self) -> None:
        self._repopulate()

    def action_back_to_projects(self) -> None:
        as_remote_app(self.app).selected_project = None
        self.app.pop_screen()

    def action_show_runs(self) -> None:
        self._highlight_nav(_NAV_RUNS)
        self._select_section("runs")

    def action_show_triggers(self) -> None:
        self._highlight_nav(_NAV_TRIGGERS)
        self._select_section("triggers")

    def action_show_tasks(self) -> None:
        self._highlight_nav(_NAV_TASKS)
        self._select_section("tasks")

    def action_show_apps(self) -> None:
        self._highlight_nav(_NAV_APPS)
        self._select_section("apps")

    def _highlight_nav(self, item_id: str) -> None:
        sidebar = self.query_one("#project-sidebar", ListView)
        for idx, child in enumerate(sidebar.children):
            if child.id == item_id:
                sidebar.index = idx
                break

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id
        if item_id == _NAV_RUNS:
            self._select_section("runs")
        elif item_id == _NAV_TRIGGERS:
            self._select_section("triggers")
        elif item_id == _NAV_TASKS:
            self._select_section("tasks")
        elif item_id == _NAV_APPS:
            self._select_section("apps")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "status-filter":
            self._repopulate()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "section-filter":
            self._repopulate()

    def action_open_detail(self) -> None:
        table = self.query_one("#hub-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        key = str(row_key.value)
        if key == "error":
            return
        if self._section == "runs":
            self.app.push_screen(RunDetailScreen(key))
        elif self._section == "triggers" and "/" in key:
            task_name, name = key.split("/", 1)
            self.app.push_screen(EntityDetailScreen("Trigger", name, task_name=task_name))
        elif self._section == "tasks":
            self.app.push_screen(EntityDetailScreen("Task", key))
        elif self._section == "apps":
            self.app.push_screen(EntityDetailScreen("App", key))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = str(event.row_key.value)
        if self._section == "runs":
            self.app.push_screen(RunDetailScreen(key))
        elif self._section == "triggers" and "/" in key:
            task_name, name = key.split("/", 1)
            self.app.push_screen(EntityDetailScreen("Trigger", name, task_name=task_name))
        elif self._section == "tasks":
            self.app.push_screen(EntityDetailScreen("Task", key))
        elif self._section == "apps":
            self.app.push_screen(EntityDetailScreen("App", key))

    def on_resize(self, event: Resize) -> None:
        self._repopulate()


class EntityDetailScreen(Screen):
    """Read-only detail for Task / App / Trigger."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "go_back", "Back"),
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

                apps = list_apps(limit=500)
                app = next((a for a in apps if a.name == self._name), None)
                if app is None:
                    lines.append(f"App '{self._name}' not found.")
                else:
                    lines.append(f"name:       {app.name}")
                    lines.append(f"status:     {_format_app_deployment_status(app.deployment_status)}")
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
        app = as_remote_app(self.app)
        try:
            scope = list_scope(app)
            actions = list_actions_for_run(self._run_name, **scope)
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
            lines = fetch_log_tail(self._run_name, action_name, max_lines=300, show_ts=True)
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
            self._reload_run_data()
        except Exception as exc:
            self.notify(f"Abort failed: {exc}", severity="error")
