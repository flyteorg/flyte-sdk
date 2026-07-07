"""Main Textual application for the remote Flyte TUI."""

from __future__ import annotations

import os
import threading
import time
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Footer, Header

from ._client import ClusterContext, init_cluster
from ._settings import resolve_config_key
from ._styles import APP_CSS

_FORCE_EXIT_DELAY_S = 0.5


def _schedule_force_exit(delay: float = _FORCE_EXIT_DELAY_S) -> None:
    """Hard-exit if Textual shutdown is blocked by uncancellable thread workers.

    Background reload/log workers call synchronous gRPC via ``run_in_executor``.
    Cancelling the worker task does not interrupt the thread, so Python can remain
    alive until every in-flight request completes (often tens of seconds).
    """

    def _force_exit() -> None:
        time.sleep(delay)
        os._exit(0)

    threading.Thread(target=_force_exit, daemon=True).start()


class RemoteTUIApp(App[None]):
    """Browse a remote Flyte v2 cluster (Devbox UI layout)."""

    CSS = APP_CSS

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        config: str | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self._config = config
        self.config_key = resolve_config_key(config)
        self.poll_interval = poll_interval
        self.cluster: ClusterContext | None = None
        self.selected_project: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Flyte"
        try:
            self.cluster = init_cluster(config=self._config)
        except Exception as exc:
            self.notify(f"Failed to connect: {exc}", severity="error", timeout=10)
            self.exit(return_code=1)
            return
        domain = self.cluster.domain
        ep = self.cluster.endpoint or "(configured)"
        self.sub_title = f"{domain} @ {ep}"
        from ._screens import ProjectsScreen

        self.push_screen(ProjectsScreen())

    def set_subtitle_for_project(self, project: str) -> None:
        if self.cluster is None:
            return
        ep = self.cluster.endpoint or ""
        self.sub_title = f"{self.cluster.domain} / {project}" + (f" @ {ep}" if ep else "")

    async def action_quit(self) -> None:
        self.workers.cancel_all()
        self.exit()
        _schedule_force_exit()
