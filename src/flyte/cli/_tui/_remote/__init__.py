"""Remote Flyte cluster TUI.

Browse and interact with a remote Flyte v2 cluster: projects, runs, actions,
logs, tasks, apps, and triggers. Launched via ``flyte start tui`` when the
resolved config points at a remote endpoint. See ``flyte.cli._tui`` for the
launcher and mode detection.
"""

from __future__ import annotations
