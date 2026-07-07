"""TUI session context (selected project / domain)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from textual.app import App

    from ._app import RemoteTUIApp


def as_remote_app(app: App) -> RemoteTUIApp:
    """Return the running ``RemoteTUIApp`` from a Textual ``App`` reference."""
    from ._app import RemoteTUIApp

    return cast(RemoteTUIApp, app)


def require_project(app: RemoteTUIApp) -> str:
    if not app.selected_project:
        raise RuntimeError("No project selected")
    return app.selected_project


def list_scope(app: RemoteTUIApp) -> dict[str, str]:
    """Keyword args for ``flyte.remote`` list APIs scoped to the active project."""
    if app.cluster is None:
        raise RuntimeError("Cluster not initialized")
    return {
        "project": require_project(app),
        "domain": app.cluster.domain,
    }
