"""Remote Flyte cluster TUI plugin."""

from __future__ import annotations

__all__ = ["launch_remote_tui"]

__version__ = "0.1.0"


def launch_remote_tui(
    *,
    project: str | None = None,
    domain: str | None = None,
    poll_interval: float = 2.0,
) -> None:
    """Launch the remote cluster TUI."""
    try:
        from ._app import RemoteTUIApp
    except ImportError as exc:
        raise ImportError(
            "The remote TUI requires the 'textual' package. Install with:\n"
            "  pip install flyteplugins-remote-tui\n"
            "  # or: pip install flyte[tui]"
        ) from exc

    app = RemoteTUIApp(project=project, domain=domain, poll_interval=poll_interval)
    app.run()
