"""Textual screens mirroring the Flyte 2 Devbox UI hierarchy."""

from __future__ import annotations

import datetime
import webbrowser
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
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

from flyte.models import ActionPhase

from .._app import ActionTreeWidget, ConditionInputPanel, DetailPanel
from .._explore import StatusSelect, _fmt_duration, _fmt_time
from .._tracker import ActionTracker
from ._client import (
    MIN_PAGE_SIZE,
    PagedResult,
    abort_run,
    activate_project,
    fetch_log_tail,
    get_run,
    list_actions_for_run,
    list_apps_paginated,
    list_projects,
    list_runs_paginated,
    list_tasks_paginated,
    list_triggers_paginated,
    signal_condition_action,
)
from ._context import as_remote_app, list_scope
from ._settings import get_recent_projects, record_recent_project
from ._sync import load_run_into_tracker

if TYPE_CHECKING:
    from textual.timer import Timer

    import flyte.remote as remote


def _call_from_thread(screen: Screen, fn, *args, **kwargs) -> None:
    """Run ``fn(*args, **kwargs)`` on the UI thread; no-op if the app is shutting down.

    Background (thread) workers cannot be interrupted, so a worker may still be
    running after the app starts to exit. ``call_from_thread`` raises in that
    window, which we swallow.
    """
    try:
        screen.app.call_from_thread(lambda: fn(*args, **kwargs))
    except Exception:
        pass


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
    "paused": "yellow",
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
        ActionPhase.PAUSED,
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

_DOCS_URL = "https://www.union.ai/docs/v2/flyte/user-guide/"


class _DocsLink(Static, can_focus=True):
    """Clickable, focusable docs link rendered at the bottom of the sidebar."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "open", "Open Docs", show=False),
    ]

    def __init__(self, url: str, label: str = "Documentation", **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._url = url

    def on_click(self) -> None:
        self.action_open()

    def action_open(self) -> None:
        webbrowser.open(self._url)


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
    if phase == "succeeded":
        return "✓"
    if phase in ("failed", "aborted", "timed_out"):
        return "✗"
    if phase == "paused":
        return "⏸"
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

    def __init__(self) -> None:
        super().__init__()
        # Cache the fetched project list so filtering/resize re-render locally
        # instead of re-hitting the network.
        self._projects: list[remote.Project] | None = None
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="projects-content"):
            with Vertical(id="recent-sidebar"):
                yield ListView(id="recent-projects")
                with Vertical(id="sidebar-footer"):
                    yield _DocsLink(_DOCS_URL, id="docs-link")
            with Vertical(id="projects-main"):
                with Horizontal(id="filter-bar"):
                    yield Label("Search:")
                    yield Input(placeholder="Filter projects...", id="project-search")
                yield EntityTable(id="projects-table", classes="EntityTable")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Projects"
        table = self.query_one("#projects-table", EntityTable)
        table.cursor_type = "row"
        self._reset_columns(table)
        table.add_row("Loading projects…", "", "")
        table.focus()
        self.query_one("#recent-sidebar", Vertical).border_title = "Recent"
        self._load_projects()

    @staticmethod
    def _reset_columns(table: EntityTable) -> None:
        table.clear(columns=True)
        table.add_column("Name", width=28)
        table.add_column("Labels", width=24)
        table.add_column("ID", width=28)

    @work(thread=True, exclusive=True)
    def _load_projects(self) -> None:
        """Connect (if needed) and fetch projects off the UI thread."""
        try:
            as_remote_app(self.app).ensure_connected()
            projects = list_projects()
        except Exception as exc:
            _call_from_thread(self, self._on_projects_loaded, None, str(exc))
            return
        _call_from_thread(self, self._on_projects_loaded, projects, None)

    def _on_projects_loaded(self, projects: list | None, error: str | None) -> None:
        app = as_remote_app(self.app)
        if error is None and app.cluster is not None:
            ep = app.cluster.endpoint or "(configured)"
            app.sub_title = f"{app.cluster.domain} @ {ep}"
        self._projects = projects
        self._error = error
        self.run_worker(self._render_projects(), group="render", exclusive=True)

    async def _render_projects(self) -> None:
        """Render the cached project list (no network) into the table + recents."""
        table = self.query_one("#projects-table", EntityTable)
        search = self.query_one("#project-search", Input).value.strip().lower()
        sidebar = self.query_one("#recent-sidebar", Vertical)
        recent_list = self.query_one("#recent-projects", ListView)
        self._reset_columns(table)
        if self._error is not None:
            await recent_list.clear()
            sidebar.border_title = "Recent"
            table.add_row(f"Error: {self._error}", "", "")
            self.sub_title = "error"
            return
        projects = self._projects or []
        by_id: dict[str, remote.Project] = {proj.pb2.id: proj for proj in projects}
        await self._populate_recent_projects(by_id, search)
        count = 0
        for proj in projects:
            name = proj.pb2.name or proj.pb2.id
            if search and search not in name.lower() and search not in proj.pb2.id.lower():
                continue
            count += 1
            table.add_row(name, _format_labels(proj), proj.pb2.id, key=proj.pb2.id)
        self.sub_title = f"{count} total"

    async def _populate_recent_projects(self, by_id: dict[str, remote.Project], search: str) -> None:
        sidebar = self.query_one("#recent-sidebar", Vertical)
        recent_list = self.query_one("#recent-projects", ListView)
        # ListView.clear() is async (queues a remove); await it so prior ListItem
        # widgets are fully detached before re-appending items with the same IDs.
        await recent_list.clear()
        app = as_remote_app(self.app)
        recent_ids = get_recent_projects(app.config_key)
        shown: list[str] = []
        seen: set[str] = set()
        for project_id in recent_ids:
            if project_id in seen:
                continue
            proj = by_id.get(project_id)
            if proj is None:
                continue
            name = proj.pb2.name or proj.pb2.id
            if search and search not in name.lower() and search not in project_id.lower():
                continue
            seen.add(project_id)
            shown.append(project_id)
            recent_list.append(ListItem(Static(name), id=f"recent-{project_id}"))
        sidebar.border_title = f"Recent ({len(shown)})" if shown else "Recent"

    def action_refresh(self) -> None:
        table = self.query_one("#projects-table", EntityTable)
        self._reset_columns(table)
        table.add_row("Loading projects…", "", "")
        self._load_projects()

    def _open_project(self, project_id: str) -> None:
        app = as_remote_app(self.app)
        cluster = app.cluster
        if cluster is None:
            return
        # activate_project re-inits the client (blocking gRPC); do it off the UI
        # thread and push the hub screen once the scope is active.
        self.sub_title = f"opening {project_id}…"
        self._activate_and_open(project_id, cluster.domain, cluster.org)

    @work(thread=True, exclusive=True)
    def _activate_and_open(self, project_id: str, domain: str, org: str) -> None:
        app = as_remote_app(self.app)
        try:
            activate_project(config=app._config, project=project_id, domain=domain, org=org)
        except Exception as exc:
            _call_from_thread(self, self._notify_error, f"Failed to open {project_id}: {exc}")
            return
        _call_from_thread(self, self._push_hub, project_id)

    def _notify_error(self, message: str) -> None:
        self.notify(message, severity="error", timeout=8)
        self.sub_title = "error"

    def _push_hub(self, project_id: str) -> None:
        app = as_remote_app(self.app)
        record_recent_project(app.config_key, project_id)
        app.selected_project = project_id
        app.set_subtitle_for_project(project_id)
        self.app.push_screen(ProjectHubScreen(project_id))

    def action_open_project(self) -> None:
        table = self.query_one("#projects-table", EntityTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self._open_project(str(row_key.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Filter the already-loaded list locally (no network) when Enter is pressed.
        if event.input.id == "project-search":
            self.run_worker(self._render_projects(), group="render", exclusive=True)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_project(str(event.row_key.value))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("recent-"):
            self._open_project(item_id.removeprefix("recent-"))


class ProjectHubScreen(Screen):
    """Project workspace: sidebar (Runs / Triggers / Tasks / Apps) + list."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back_to_projects", "Projects"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "open_detail", "Open"),
        Binding("[", "prev_page", "Previous page", show=False),
        Binding("]", "next_page", "Next page", show=False),
        Binding("1", "show_runs", "Runs", show=False),
        Binding("2", "show_triggers", "Triggers", show=False),
        Binding("3", "show_tasks", "Tasks", show=False),
        Binding("4", "show_apps", "Apps", show=False),
    ]

    def __init__(self, project_name: str) -> None:
        super().__init__()
        self._project = project_name
        self._section = "runs"
        self._page = 0
        self._has_next = False
        self._page_size = MIN_PAGE_SIZE

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
            with Vertical(id="hub-content"):
                with Horizontal(id="hub-header"):
                    yield Static("", id="section-title")
                    yield Static("", id="page-indicator")
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

    def _reset_page(self) -> None:
        self._page = 0

    def _effective_page_size(self) -> int:
        """Fit page size to visible table rows (header/footer/chrome excluded)."""
        # Prefer the table's actual rendered height when available; this naturally
        # accounts for header, footer, hub header, filter bar, and any padding,
        # since the table is laid out as ``height: 1fr``.
        try:
            table_height = self.query_one("#hub-table", EntityTable).size.height
        except Exception:
            table_height = 0
        if table_height > 1:
            # DataTable reserves one row for its column header.
            return max(MIN_PAGE_SIZE, table_height - 1)
        # Fallback used before the table is mounted/sized on first paint.
        # Overhead = screen header + footer + hub header + filter bar (3 rows) + table column header.
        overhead = 7
        return max(MIN_PAGE_SIZE, self.size.height - overhead)

    def _update_page_indicator(self, paged: PagedResult | None = None) -> None:
        indicator = self.query_one("#page-indicator", Static)
        if paged is None or not paged.items:
            if self._page == 0 and not self._has_next:
                indicator.update("")
                return
        parts = [f"page {self._page + 1}"]
        if paged and paged.items:
            start = self._page * self._page_size + 1
            end = self._page * self._page_size + len(paged.items)
            parts.append(f"{start}-{end}")
        hints: list[str] = []
        if self._page > 0:
            hints.append("[")
        if self._has_next:
            hints.append("]")
        if hints:
            parts.append("".join(hints))
        indicator.update(" ".join(parts))

    def on_resize(self, event: Resize) -> None:
        new_size = self._effective_page_size()
        if new_size != self._page_size:
            self._repopulate()

    def _select_section(self, section: str) -> None:
        self._section = section
        self._reset_page()
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
        # Reset any previous query before configuring the filter for this section,
        # so switching sections doesn't carry stale text into the new filter.
        section_filter.value = ""
        if section == "runs":
            status_label.display = True
            status_filter.display = True
            filter_label.display = True
            filter_label.update("Task:")
            section_filter.display = True
            section_filter.placeholder = "Filter by task name..."
        elif section == "tasks":
            status_label.display = False
            status_filter.display = False
            filter_label.display = True
            filter_label.update("Name:")
            section_filter.display = True
            section_filter.placeholder = "Filter by task name..."
        elif section == "triggers":
            status_label.display = False
            status_filter.display = False
            filter_label.display = True
            filter_label.update("Search:")
            section_filter.display = True
            section_filter.placeholder = "Filter triggers..."
        elif section == "apps":
            status_label.display = False
            status_filter.display = False
            filter_label.display = True
            filter_label.update("Name:")
            section_filter.display = True
            section_filter.placeholder = "Filter by app name..."
        else:
            status_label.display = False
            status_filter.display = False
            filter_label.display = False
            section_filter.display = False
        self._repopulate()

    def _repopulate(self) -> None:
        self._page_size = self._effective_page_size()
        section = self._section
        scope = list_scope(as_remote_app(self.app))
        filt = self.query_one("#section-filter", Input).value.strip() or None
        phase_key = str(self.query_one("#status-filter", Select).value or "all")
        self.query_one("#hub-table", EntityTable).clear(columns=True)
        self.sub_title = "loading…"
        # Fetch off the UI thread so navigation/paging/filtering stays responsive.
        self._load_section(section, self._page, self._page_size, filt, phase_key, scope)

    @work(thread=True, exclusive=True)
    def _load_section(
        self,
        section: str,
        page: int,
        page_size: int,
        filt: str | None,
        phase_key: str,
        scope: dict[str, str],
    ) -> None:
        paged: PagedResult
        try:
            if section == "runs":
                paged = list_runs_paginated(
                    page=page,
                    page_size=page_size,
                    task_name=filt,
                    in_phase=_PHASE_FILTER_MAP.get(phase_key),
                    **scope,
                )
            elif section == "triggers":
                paged = list_triggers_paginated(page=page, page_size=page_size, search=filt)
            elif section == "tasks":
                paged = list_tasks_paginated(page=page, page_size=page_size, task_name=filt, **scope)
            elif section == "apps":
                paged = list_apps_paginated(page=page, page_size=page_size, name=filt)
            else:
                return
        except Exception as exc:
            _call_from_thread(self, self._render_section_error, section, str(exc))
            return
        _call_from_thread(self, self._render_section, section, paged)

    def _render_section(self, section: str, paged: PagedResult) -> None:
        # Ignore a stale fetch if the user switched sections while it was in flight.
        if section != self._section:
            return
        table = self.query_one("#hub-table", EntityTable)
        table.clear(columns=True)
        self._has_next = paged.has_next
        self._update_page_indicator(paged)
        self.sub_title = f"{len(paged.items)} on page"
        if section == "runs":
            self._render_runs(table, paged)
        elif section == "triggers":
            self._render_triggers(table, paged)
        elif section == "tasks":
            self._render_tasks(table, paged)
        elif section == "apps":
            self._render_apps(table, paged)

    def _render_section_error(self, section: str, message: str) -> None:
        if section != self._section:
            return
        table = self.query_one("#hub-table", EntityTable)
        table.clear(columns=True)
        table.add_column("Error")
        self._has_next = False
        self._update_page_indicator()
        table.add_row(f"Error: {message}")
        self.sub_title = "error"

    def _render_runs(self, table: EntityTable, paged: PagedResult) -> None:
        table.add_column("", width=3)
        table.add_column("Run", width=22)
        table.add_column("Task", width=18)
        table.add_column("Duration", width=10)
        table.add_column("Started", width=14)
        table.add_column("Ended", width=14)
        for run in paged.items:
            phase = str(run.phase).lower() if hasattr(run.phase, "value") else str(run.phase)
            icon = Text(_phase_icon(phase), style=_STATUS_COLORS.get(phase, ""))
            task = run.action.task_name or "-"
            st = run.action.start_time
            started = _fmt_time(st.timestamp())
            ended = ""
            if run.action.done() and run.action.pb2.status.HasField("end_time"):
                end = run.action.pb2.status.end_time.ToDatetime()
                ended = _fmt_time(end.timestamp())
            table.add_row(icon, run.name, task, _run_duration(run), started, ended, key=run.name)

    def _render_triggers(self, table: EntityTable, paged: PagedResult) -> None:
        table.add_column("Name", width=24)
        table.add_column("Task", width=24)
        table.add_column("Active", width=10)
        for tr in paged.items:
            active = "yes" if tr.is_active else "no"
            table.add_row(tr.name, tr.task_name, active, key=f"{tr.task_name}/{tr.name}")

    def _render_tasks(self, table: EntityTable, paged: PagedResult) -> None:
        table.add_column("Name", width=28)
        table.add_column("Version", width=14)
        table.add_column("Short name", width=18)
        table.add_column("Env", width=18)
        for t in paged.items:
            meta = t.pb2.metadata
            # Proto3 string fields have no presence; read values directly (empty string if unset).
            short = meta.short_name or "-"
            env = meta.environment_name or "-"
            table.add_row(t.name, t.version, short, env, key=t.name)

    def _render_apps(self, table: EntityTable, paged: PagedResult) -> None:
        table.add_column("Name", width=24)
        table.add_column("Status", width=18)
        table.add_column("Endpoint", width=36)
        for app in paged.items:
            status = _format_app_deployment_status(app.deployment_status).lower()
            table.add_row(app.name, status, app.endpoint or "", key=app.name)

    def action_refresh(self) -> None:
        self._repopulate()

    def action_prev_page(self) -> None:
        if self._page == 0:
            return
        self._page -= 1
        self._repopulate()

    def action_next_page(self) -> None:
        if not self._has_next:
            return
        self._page += 1
        self._repopulate()

    def action_back_to_projects(self) -> None:
        as_remote_app(self.app).selected_project = None
        self.app.pop_screen()
        screen = self.app.screen
        if isinstance(screen, ProjectsScreen):
            # Re-render from the cached project list (updates the recents panel);
            # no network fetch needed.
            screen.run_worker(screen._render_projects(), group="render", exclusive=True)

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
            self._reset_page()
            self._repopulate()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # The section filter triggers a (potentially expensive) server fetch, so wait
        # for the user to press Enter rather than firing on every keystroke.
        if event.input.id == "section-filter":
            self._reset_page()
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


class EntityDetailScreen(Screen):
    """Read-only detail for Task / App / Trigger."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, kind: str, name: str, *, task_name: str | None = None) -> None:
        super().__init__()
        self._kind = kind
        self._entity_name = name
        self._task_name = task_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(id="detail-body"), id="detail-scroll")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"{self._kind}: {self._entity_name}"
        self.query_one("#detail-body", Static).update("Loading…")
        self._load_detail()

    @work(thread=True, exclusive=True)
    def _load_detail(self) -> None:
        """Fetch the entity detail off the UI thread, then render the text."""
        try:
            lines = self._fetch_detail_lines()
        except Exception as exc:
            lines = [f"Error loading {self._kind}: {exc}"]
        _call_from_thread(self, self._set_body, "\n".join(lines))

    def _fetch_detail_lines(self) -> list[str]:
        import flyte.remote as remote

        lines: list[str] = []
        if self._kind == "Task":
            t = remote.Task.get(name=self._entity_name, version="latest")
            details = t.fetch()
            lines.append(f"name:       {self._entity_name}")
            lines.append(f"version:    {details.pb2.id.version}")  # ty: ignore[unresolved-attribute]
            lines.append(f"type:       {details.pb2.task_template.type}")  # ty: ignore[unresolved-attribute]
            lines.append(f"deployed:   {details.pb2.metadata.deployed_at.ToDatetime()}")
        elif self._kind == "App":
            app_obj = remote.App.get(name=self._entity_name)
            lines.append(f"name:       {app_obj.name}")
            lines.append(f"status:     {_format_app_deployment_status(app_obj.deployment_status)}")
            lines.append(f"endpoint:   {app_obj.endpoint or '(none)'}")
            lines.append(f"url:        {app_obj.url}")
        elif self._kind == "Trigger":
            assert self._task_name
            tr = remote.Trigger.get(name=self._entity_name, task_name=self._task_name)
            lines.append(f"name:       {tr.name}")
            lines.append(f"task:       {tr.task_name}")
            lines.append(f"active:     {tr.is_active}")
        return lines

    def _set_body(self, text: str) -> None:
        self.query_one("#detail-body", Static).update(text)

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
        self._selected_action: str | None = None
        self._seen_pending_condition_ids: set[str] = set()
        self._actions_by_name: dict[str, remote.Action] = {}
        self._active = True
        self._poll_timer: Timer | None = None

    def _is_active(self) -> bool:
        return self._active and not getattr(self.app, "_exit", False)

    def on_unmount(self) -> None:
        self._active = False
        self.workers.cancel_node(self)
        if self._poll_timer is not None:
            self._poll_timer.stop()

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
        self._poll_timer = self.set_interval(poll, self._poll_run)
        self._reload_run_data(fetch_io=True)
        self._refresh_logs()

    @work(thread=True, exclusive=True)
    def _reload_run_data(self, fetch_io: bool = True) -> None:
        if not self._is_active():
            return
        app = as_remote_app(self.app)
        try:
            scope = list_scope(app)
            actions = list_actions_for_run(self._run_name, **scope)
            if not self._is_active():
                return
            self._actions_by_name = {a.name: a for a in actions}
            load_run_into_tracker(self._tracker, actions, fetch_io=fetch_io)
            if not self._is_active():
                return
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
            if self._is_active():
                self.app.call_from_thread(self._apply_run_error, str(exc))

    def _apply_run_data(self, info_lines: list[str], phase: str) -> None:
        if not self._is_active():
            return
        try:
            self.query_one("#run-info-body", Static).update("\n".join(info_lines))
            tree = self.query_one("#action-tree", ActionTreeWidget)
            tree.refresh_from_tracker()
            detail = self.query_one("#detail-panel", DetailPanel)
            for pc in self._tracker.get_all_pending_conditions():
                if pc.action_id not in self._seen_pending_condition_ids:
                    self._seen_pending_condition_ids.add(pc.action_id)
                    if tree.focus_action(pc.action_id):
                        detail.action_id = pc.action_id
                    self.query_one("#right-tabs", TabbedContent).active = "tab-details"
            detail.refresh_detail()
            self.sub_title = phase
        except Exception:
            pass

    def _apply_run_error(self, message: str) -> None:
        self.sub_title = f"error: {message}"

    def _poll_run(self) -> None:
        if not self._is_active():
            return
        try:
            run = get_run(self._run_name)
            if not run.done():
                # Poll only refreshes action phases; skip per-action I/O fetches.
                self._reload_run_data(fetch_io=False)
        except Exception:
            pass

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _refresh_logs(self) -> None:
        if not self._is_active():
            return
        action_name = self._selected_action
        cached_action = self._actions_by_name.get(action_name) if action_name else None
        try:
            lines = fetch_log_tail(
                self._run_name,
                action_name,
                action=cached_action,
                max_lines=300,
                show_ts=True,
                should_continue=self._is_active,
            )
            if not self._is_active():
                return
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

    @on(ConditionInputPanel.Submitted)
    def _on_condition_submitted(self, event: ConditionInputPanel.Submitted) -> None:
        self._signal_condition(event.action_id, event.value)

    @work(thread=True, exclusive=True)
    def _signal_condition(self, action_id: str, value: bool | int | float | str) -> None:
        if not self._is_active():
            return
        try:
            signal_condition_action(self._run_name, action_id, value)
            self.app.call_from_thread(self.notify, "Condition signaled")
            self.app.call_from_thread(self._reload_run_data)
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Signal failed: {exc}",
                severity="error",
            )

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
        self.notify("Aborting run…")
        self._abort_run()

    @work(thread=True, exclusive=True)
    def _abort_run(self) -> None:
        if not self._is_active():
            return
        try:
            abort_run(self._run_name)
        except Exception as exc:
            _call_from_thread(self, self.notify, f"Abort failed: {exc}", severity="error")
            return
        _call_from_thread(self, self.notify, "Run abort requested")
        self._reload_run_data()
