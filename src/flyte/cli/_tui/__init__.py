from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from ._tracker import ActionTracker

_TUI_IMPORT_HINT = "The TUI requires the 'textual' package. Install it with:  pip install flyte[tui]"


def launch_tui(tracker: ActionTracker, execute_fn: Callable[[], Awaitable[Any]]) -> None:
    """Launch the interactive TUI.

    Raises a helpful `ImportError` when `textual` is not installed.
    """
    try:
        from ._app import FlyteTUIApp
    except ImportError as exc:
        raise ImportError(_TUI_IMPORT_HINT) from exc

    app = FlyteTUIApp(tracker=tracker, execute_fn=execute_fn)
    app.run()


def launch_tui_explore() -> None:
    """Launch the TUI explore mode to browse past local runs.

    Raises a helpful `ImportError` when `textual` is not installed.
    """
    try:
        from ._explore import ExploreTUIApp
    except ImportError as exc:
        raise ImportError(_TUI_IMPORT_HINT) from exc

    app = ExploreTUIApp()
    app.run()


def launch_tui_remote(config: str | None = None, poll_interval: float = 2.0) -> None:
    """Launch the remote cluster TUI to browse a remote Flyte v2 cluster.

    Raises a helpful `ImportError` when `textual` is not installed.
    """
    try:
        from ._remote._app import RemoteTUIApp
    except ImportError as exc:
        raise ImportError(_TUI_IMPORT_HINT) from exc

    app = RemoteTUIApp(config=config, poll_interval=poll_interval)
    app.run()


def config_is_remote(config: str | None = None) -> bool:
    """Return True when the resolved config targets a remote Flyte cluster.

    A remote target is detected when the config resolves a platform endpoint or
    a ``FLYTE_API_KEY`` is set. Otherwise the config is treated as local and the
    explore TUI is used to browse persisted local runs.
    """
    if os.getenv("FLYTE_API_KEY"):
        return True
    import flyte.config as flyte_config

    cfg = flyte_config.auto(config)
    return bool(cfg.platform.endpoint)
