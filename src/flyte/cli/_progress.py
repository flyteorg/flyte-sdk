"""
A live, rich-powered display for local uploads driven by the CLI.

`flyte create artifact` can push a multi-gigabyte file, which is a long time to sit
behind one opaque spinner. This renders a panel with a bar per file and phase -- the
md5 pass over the file, then the PUT to blob storage -- with size, throughput and ETA:

    ╭─ Publishing artifact my_model ─────────────────────────────────────────────────╮
    │ ✔ uploaded  model_card.html   ━━━━━━━━━━━━━━━━ 100% 12.1/12.1 KiB        00:00 │
    │ ✔ hashed    model.pt          ━━━━━━━━━━━━━━━━ 100% 2.1/2.1 GiB          00:07 │
    │ ⠹ uploading model.pt          ━━━━━━━━━╸        62% 1.3/2.1 GiB 18.7 MB/s 00:48 │
    │ ⠹ publishing model.pt                                                          │
    ╰──────────────────────────────────────────────────────── testing/development ───╯

Anything non-interactive (piped output, CI, `--no-progress`) gets a plain-text fallback,
since `get_console()` forces a terminal and would otherwise emit animation frames into
a log file.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.spinner import Spinner
from rich.table import Column, Table
from rich.text import Text

from flyte.cli._common import PREFERRED_BORDER_COLOR, get_console, safe_spinner
from flyte.remote import _progress as upload_progress

# Flyte purple, matching the TUI palette.
_ACCENT = "#7652a2"
_DONE = "green"

# Past tense for a finished phase, so a completed row doesn't read as still running.
_PHASE_DONE_LABELS: Dict[str, str] = {"hashing": "hashed", "uploading": "uploaded"}

# Files smaller than this hash faster than a frame refresh, so their hashing row would
# only ever be seen already-complete. Skip it and let the upload row speak for them.
_HASH_ROW_MIN_BYTES = 64 * 1024 * 1024


@dataclass
class _Entry:
    """What the display tracks per reported key; `task_id` is None for skipped rows."""

    task_id: Optional[TaskID]
    phase: str
    total: int
    completed: int = 0


class _SpeedColumn(TransferSpeedColumn):
    """TransferSpeedColumn, minus the red '?' it shows for an unmeasurable transfer."""

    def render(self, task) -> Text:
        if task.finished or task.speed is None:
            return Text("")
        return super().render(task)


class _PhaseColumn(ProgressColumn):
    """The verb for a row: dim while it runs, red once it has failed."""

    def render(self, task) -> Text:
        phase = str(task.fields.get("phase", ""))
        return Text(phase, style="red" if phase == "failed" else "dim")


class _StatusColumn(SpinnerColumn):
    """Spinner while running, green check when done, red cross when it failed."""

    def render(self, task):
        if task.fields.get("phase") == "failed":
            return Text("✗", style="red")
        return super().render(task)


class UploadDisplay(Protocol):
    """The handle a command uses to annotate the display while work is in flight."""

    def note(self, message: str) -> None:
        """Set the status line shown underneath the bars."""


class _PlainDisplay:
    """Fallback for non-TTY output: print each note once, no animation, no bars."""

    def __init__(self, console: Console, title: str):
        self._console = console
        self._console.print(Text.from_markup(title))

    def note(self, message: str) -> None:
        self._console.print(Text.from_markup(f"[dim]{message}[/dim]"))


class _LiveUploadDisplay:
    """
    Renders upload progress into a panel and implements `UploadProgressHandler`.

    Progress callbacks arrive on syncify's background event loop thread while Rich's
    refresh thread renders, so mutations of the key -> task map are locked. Rich's own
    task table is already internally locked.
    """

    def __init__(self, console: Console, title: str, subtitle: Optional[str] = None):
        self._title = title
        self._subtitle = subtitle
        self._spinner_name = safe_spinner()
        self._progress = Progress(
            _StatusColumn(spinner_name=self._spinner_name, style=_ACCENT, finished_text=Text("✔", style=_DONE)),
            _PhaseColumn(table_column=Column(no_wrap=True, min_width=9)),
            # Fixed width: a later, longer file name must not resize the panel mid-render.
            TextColumn(
                "[bold]{task.fields[name]}[/bold]",
                table_column=Column(no_wrap=True, overflow="ellipsis", min_width=26, max_width=26),
            ),
            BarColumn(bar_width=26, complete_style=_ACCENT, finished_style=_DONE, pulse_style=_ACCENT),
            TaskProgressColumn(),
            DownloadColumn(binary_units=True),
            _SpeedColumn(),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            console=console,
            # This Progress is rendered by the Live below rather than driving its own.
            auto_refresh=False,
        )
        self._note_spinner = Spinner(self._spinner_name, style=_ACCENT)
        self._note: Optional[str] = None
        self._lock = threading.Lock()
        self._entries: Dict[str, _Entry] = {}
        self._live = Live(
            console=console,
            get_renderable=self._render,
            refresh_per_second=12,
            transient=False,
        )

    # -- rendering ---------------------------------------------------------------

    def _render(self) -> RenderableType:
        body: List[RenderableType] = []
        if self._progress.task_ids:
            body.append(self._progress)
        note = self._note
        if note:
            # The Spinner renderable animates on its own as Live re-renders it.
            line = Table.grid(padding=(0, 1))
            line.add_column()
            line.add_column()
            line.add_row(self._note_spinner, Text(note, style="dim"))
            body.append(line)
        if not body:
            body.append(Text("working…", style="dim"))
        return Panel(
            Group(*body),
            title=Text.from_markup(self._title),
            title_align="left",
            subtitle=Text.from_markup(f"[dim]{self._subtitle}[/dim]") if self._subtitle else None,
            subtitle_align="right",
            border_style=PREFERRED_BORDER_COLOR,
            padding=(0, 1),
            expand=False,
        )

    # -- public handle -----------------------------------------------------------

    def note(self, message: str) -> None:
        self._note = message

    # -- UploadProgressHandler ---------------------------------------------------

    def start(self, key: str, *, name: str, phase: str, total: int) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.task_id is not None:
                # A retried upload restarts its bar rather than stacking a second one.
                self._progress.reset(entry.task_id, total=total or None, name=name, phase=phase)
                self._entries[key] = _Entry(entry.task_id, phase=phase, total=total)
                return
            if phase == "hashing" and total < _HASH_ROW_MIN_BYTES:
                self._entries[key] = _Entry(None, phase=phase, total=total)
                return
            task_id = self._progress.add_task("", total=total or None, name=name, phase=phase)
            self._entries[key] = _Entry(task_id, phase=phase, total=total)

    def advance(self, key: str, size: int) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.completed += size
            task_id = entry.task_id
        if task_id is not None:
            self._progress.advance(task_id, size)

    def finish(self, key: str, *, failed: bool = False) -> None:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None or entry.task_id is None:
            return
        if failed:
            self._progress.update(entry.task_id, phase="failed")
            return
        # Snap to complete: a file whose size we couldn't read up front has no total, and
        # would otherwise be left rendering as a forever-pulsing bar.
        final = entry.total or entry.completed or 1
        self._progress.update(
            entry.task_id,
            completed=final,
            total=final,
            phase=_PHASE_DONE_LABELS.get(entry.phase, entry.phase),
        )

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "_LiveUploadDisplay":
        self._live.start(refresh=True)
        return self

    def __exit__(self, *exc_info) -> None:
        self._note = None
        self._live.stop()


@contextmanager
def upload_display(
    title: str,
    *,
    subtitle: Optional[str] = None,
    no_progress: bool = False,
    console: Optional[Console] = None,
) -> Iterator[UploadDisplay]:
    """
    Show live progress bars for any upload the SDK performs inside the block.

    Args:
        title: Panel title, may contain Rich markup.
        subtitle: Optional right-aligned panel subtitle (e.g. the target project/domain).
        no_progress: Honour the global `--no-progress` flag by falling back to plain text.
        console: Console to render on; defaults to the CLI's console.
    """
    console = console or get_console()
    if no_progress or not sys.stdout.isatty():
        yield _PlainDisplay(console, title)
        return

    display = _LiveUploadDisplay(console, title, subtitle)
    with display, upload_progress.report_uploads(display):
        yield display
