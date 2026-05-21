"""Main Textual application for the remote Flyte TUI."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Footer, Header

from ._client import ClusterContext, init_cluster
from ._screens import AppsScreen, RunsScreen, TasksScreen, TriggersScreen
from ._styles import APP_CSS


class RemoteTUIApp(App[None]):
    """Browse a remote Flyte v2 cluster (runs, tasks, apps, triggers)."""

    CSS = APP_CSS

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("1", "show_runs", "Runs", show=False),
        Binding("2", "show_tasks", "Tasks", show=False),
        Binding("3", "show_apps", "Apps", show=False),
        Binding("4", "show_triggers", "Triggers", show=False),
    ]

    def __init__(
        self,
        *,
        project: str | None = None,
        domain: str | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self._project = project
        self._domain = domain
        self.poll_interval = poll_interval
        self.cluster: ClusterContext | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Flyte Remote"
        try:
            self.cluster = init_cluster(project=self._project, domain=self._domain)
        except Exception as exc:
            self.notify(f"Failed to connect: {exc}", severity="error", timeout=10)
            self.exit(1)
            return
        ep = self.cluster.endpoint or "(configured)"
        self.sub_title = f"{self.cluster.project}/{self.cluster.domain} @ {ep}"
        self._show_screen(RunsScreen)

    def _show_screen(self, screen_cls) -> None:
        for screen in list(self.screen_stack):
            if screen is not self.screen:
                self.pop_screen()
        self.push_screen(screen_cls())

    def action_show_runs(self) -> None:
        self._show_screen(RunsScreen)

    def action_show_tasks(self) -> None:
        self._show_screen(TasksScreen)

    def action_show_apps(self) -> None:
        self._show_screen(AppsScreen)

    def action_show_triggers(self) -> None:
        self._show_screen(TriggersScreen)
