from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from ._tracker import ActionTracker


def launch_tui(tracker: ActionTracker, execute_fn: Callable[[], Awaitable[Any]]) -> None:
    """Launch the interactive TUI.

    Raises a helpful ``ImportError`` when ``textual`` is not installed.
    """
    try:
        from ._app import FlyteTUIApp
    except ImportError as exc:
        raise ImportError("The TUI requires the 'textual' package. Install it with:  pip install flyte[tui]") from exc

    app = FlyteTUIApp(tracker=tracker, execute_fn=execute_fn)
    app.run()
